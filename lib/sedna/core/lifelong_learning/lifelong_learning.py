# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

import joblib

from sedna.backend import set_backend
from sedna.core.base import JobBase
from sedna.common.file_ops import FileOps
from sedna.common.constant import K8sResourceKind, K8sResourceKindStatus
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.multi_task_learning import MulTaskLearning
from sedna.service.client import KBClient


class LifelongLearning(JobBase):
    """
    Lifelong learning
    """

    def __init__(self,
                 estimator,
                 task_definition="TaskDefinitionByDataAttr",
                 task_relationship_discovery=None,
                 task_mining=None,
                 task_remodeling=None,
                 inference_integrate=None,
                 unseen_task_detect="TaskAttrFilter",

                 task_definition_param=None,
                 relationship_discovery_param=None,
                 task_mining_param=None,
                 task_remodeling_param=None,
                 inference_integrate_param=None,
                 unseen_task_detect_param=None):
        e = MulTaskLearning(
            estimator=estimator,
            task_definition=task_definition,
            task_relationship_discovery=task_relationship_discovery,
            task_mining=task_mining,
            task_remodeling=task_remodeling,
            inference_integrate=inference_integrate,
            task_definition_param=task_definition_param,
            relationship_discovery_param=relationship_discovery_param,
            task_mining_param=task_mining_param,
            task_remodeling_param=task_remodeling_param,
            inference_integrate_param=inference_integrate_param)
        self.unseen_task_detect = unseen_task_detect
        self.unseen_task_detect_param = e.parse_param(
            unseen_task_detect_param
        )
        config = dict(
            ll_kb_server=Context.get_parameters("KB_SERVER"),
            output_url=Context.get_parameters("OUTPUT_URL", "/tmp")
        )
        task_index = FileOps.join_path(config['output_url'], 'index.pkl')
        config['task_index'] = task_index
        super(LifelongLearning, self).__init__(
            estimator=e, config=config
        )
        self.job_kind = K8sResourceKind.LIFELONG_JOB.value
        self.kb_server = KBClient(kbserver=self.config.ll_kb_server)

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              action="initial",
              **kwargs):
        """
        :param train_data: data use to train model
        :param valid_data: data use to valid model
        :param post_process: callback function
        :param action: initial - kb init, update - kb incremental update
        """

        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        res = self.estimator.train(
            train_data=train_data,
            valid_data=valid_data,
            **kwargs
        )  # todo: Distinguishing incremental update and fully overwrite

        task_index_url = self.estimator.estimator.task_index_url
        task_index = joblib.load(task_index_url)

        extractor = task_index['extractor']
        task_groups = task_index['task_groups']

        for task in task_groups:
            model_file = task.model.model
            save_model = FileOps.join_path(
                self.config.output_url,
                os.path.basename(model_file)
            )
            model_file = FileOps.upload(model_file, save_model)
            try:
                model = self.kb_server.upload_file(save_model)
            except Exception as err:
                self.log.error(f"Upload task model of {model_file} fail: {err}")
                model_obj = set_backend(
                    estimator=self.estimator.estimator.base_model
                )
                model = model_obj.load(model_file)
            task.model.model = model

            for _task in task.tasks:
                sample_dir = FileOps.join_path(
                    self.config.output_url,
                    f"{_task.samples.data_type}_{_task.entry}.sample")
                task.samples.save(sample_dir)
                try:
                    sample_dir = self.kb_server.upload_file(sample_dir)
                except Exception as err:
                    self.log.error(
                        f"Upload task samples of {_task.entry} fail: {err}")
                _task.samples.data_url = sample_dir

        task_info = {
            "task_groups": task_groups,
            "extractor": extractor
        }
        fd, name = tempfile.mkstemp()
        joblib.dump(task_info, name)

        index_file = self.kb_server.update_db(name)
        if not index_file:
            self.log.error(f"KB update Fail !")
            index_file = name
        FileOps.upload(index_file, self.config.task_index)

        task_info_res = self.estimator.model_info(
            self.config.task_index, result=res,
            relpath=self.config.data_path_prefix)
        self.report_task_info(
            None, K8sResourceKindStatus.COMPLETED.value, task_info_res)
        self.log.info(f"Lifelong learning Train task Finished, "
                      f"KB idnex save in {self.config.task_index}")
        return callback_func(self.estimator, res) if callback_func else res

    def update(self, train_data, valid_data=None, post_process=None, **kwargs):
        return self.train(
            train_data=train_data,
            valid_data=valid_data,
            post_process=post_process,
            action="update",
            **kwargs
        )

    def evaluate(self, data, post_process=None, model_threshold=0.1, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        task_index_url = self.get_parameters(
            "MODEL_URLS", self.config.task_index)
        index_url = self.estimator.estimator.task_index_url
        self.log.info(
            f"Download kb index from {task_index_url} to {index_url}")
        FileOps.download(task_index_url, index_url)
        res, tasks_detail = self.estimator.evaluate(data=data, **kwargs)
        drop_tasks = []
        for detail in tasks_detail:
            scores = detail.scores
            entry = detail.entry
            self.log.info(f"{entry} scores: {scores}")
            if any(map(lambda x: float(x) < model_threshold, scores.values())):
                self.log.warn(
                    f"{entry} will not be deploy "
                    f"because scores lt {model_threshold}")
                drop_tasks.append(entry)
                continue
        drop_task = ",".join(drop_tasks)
        index_file = self.kb_server.update_task_status(drop_task, new_status=0)
        if not index_file:
            self.log.error(f"KB update Fail !")
            index_file = str(index_url)
        self.log.info(
            f"upload kb index from {index_file} to {self.config.task_index}")
        FileOps.upload(index_file, self.config.task_index)
        task_info_res = self.estimator.model_info(
            self.config.task_index, result=res,
            relpath=self.config.data_path_prefix)
        self.report_task_info(
            None,
            K8sResourceKindStatus.COMPLETED.value,
            task_info_res,
            kind="eval")
        return callback_func(res) if callback_func else res

    def inference(self, data=None, post_process=None, **kwargs):
        task_index_url = self.get_parameters(
            "MODEL_URLS", self.config.task_index)
        index_url = self.estimator.estimator.task_index_url
        FileOps.download(task_index_url, index_url)
        res, tasks = self.estimator.predict(
            data=data, post_process=post_process, **kwargs
        )

        is_unseen_task = False
        if self.unseen_task_detect:

            try:
                if callable(self.unseen_task_detect):
                    unseen_task_detect_algorithm = self.unseen_task_detect()
                else:
                    unseen_task_detect_algorithm = ClassFactory.get_cls(
                        ClassType.UTD, self.unseen_task_detect
                    )()
            except ValueError as err:
                self.log.error("Lifelong learning "
                               "Inference [UTD] : {}".format(err))
            else:
                is_unseen_task = unseen_task_detect_algorithm(
                    tasks=tasks, result=res, **self.unseen_task_detect_param
                )
        return res, is_unseen_task, tasks

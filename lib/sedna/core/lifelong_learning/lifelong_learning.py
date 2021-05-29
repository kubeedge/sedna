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
import joblib
import tempfile
from sedna.core.base import JobBase
from sedna.common.file_ops import FileOps
from sedna.common.constant import K8sResourceKind, K8sResourceKindStatus
from sedna.common.config import Context
from sedna.common.log import sednaLogger
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.multi_task_learning import MulTaskLearning
from sedna.service.client import KBClient


class LifelongLearning(JobBase):
    """
    Lifelong learning Experiment
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
                 task_relationship_discovery_param=None,
                 task_mining_param=None,
                 task_remodeling_param=None,
                 inference_integrate_param=None,
                 unseen_task_detect_param=None):
        estimator = MulTaskLearning(
            estimator=estimator,
            task_definition=task_definition,
            task_relationship_discovery=task_relationship_discovery,
            task_mining=task_mining,
            task_remodeling=task_remodeling,
            inference_integrate=inference_integrate,

            task_definition_param=task_definition_param,
            task_relationship_discovery_param=task_relationship_discovery_param,
            task_mining_param=task_mining_param,
            task_remodeling_param=task_remodeling_param,
            inference_integrate_param=inference_integrate_param
        )
        self.unseen_task_detect = unseen_task_detect
        self.unseen_task_detect_param = estimator.parse_param(unseen_task_detect_param)
        config = dict(
            ll_kb_server=Context.get_parameters("KB_SERVER"),
            output_url=Context.get_parameters("OUTPUT_URL")
        )
        task_index = FileOps.join_path(config['output_url'], 'index.pkl')
        config['task_index'] = task_index
        super(LifelongLearning, self).__init__(estimator=estimator, config=config)
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
            callback_func = ClassFactory.get_cls(ClassType.CALLBACK, post_process)
        res = self.estimator.train(
            train_data=train_data,
            valid_data=valid_data,
            **kwargs
        )  # todo: Distinguishing incremental update and fully overwrite

        task_groups = self.estimator.estimator.task_groups
        extractor_file = FileOps.join_path(
            os.path.dirname(self.estimator.estimator.task_index_url),
            "kb_extractor.pkl"
        )
        extractor_file = self.kb_server.upload_file(extractor_file)
        for task in task_groups:
            task.model.model = self.kb_server.upload_file(task.model.model)

        task_info = {
            "task_groups": task_groups,
            "extractor": extractor_file
        }
        fd, name = tempfile.mkstemp()
        joblib.dump(task_info, name)

        index_file = self.kb_server.update_db(name)
        if not index_file:
            self.log.error(f"KB update Fail !")
            index_file = name

        FileOps.download(index_file, self.config.task_index)
        if os.path.isfile(name):
            os.close(fd)
            os.remove(name)
        self._report_task_info(None, K8sResourceKindStatus.COMPLETED.value,
                               res, model=self.config.task_index)
        sednaLogger.info(f"Lifelong learning Experiment Finished, "
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
            callback_func = ClassFactory.get_cls(ClassType.CALLBACK, post_process)

        FileOps.download(self.config.task_index,
                         self.estimator.estimator.task_index_url)
        res, tasks_detail = self.estimator.evaluate(data=data, **kwargs)
        drop_tasks = []
        for detail in tasks_detail:
            scores = detail.scores
            entry = detail.entry
            if any(map(lambda x: x < model_threshold, scores)):
                sednaLogger.warn(f"{entry} will not be deploy because scores lt {model_threshold}")
                drop_tasks.append(entry)
                continue

        index_file = self.kb_server.update_task_status(drop_tasks, new_status=0)
        if not index_file:
            self.log.error(f"KB update Fail !")
        else:
            FileOps.download(index_file, self.config.task_index)
        self._report_task_info(None, K8sResourceKindStatus.COMPLETED.value,
                               res, kind="eval", model=self.config.task_index)
        return callback_func(res) if callback_func else res

    def inference(self, data=None, post_process=None, **kwargs):
        FileOps.download(self.config.task_index,
                         self.estimator.estimator.task_index_url)
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
                sednaLogger.error("Lifelong learning Experiment Inference [UTD] : {}".format(err))
            else:
                is_unseen_task = unseen_task_detect_algorithm(
                    tasks=tasks, result=res, **self.unseen_task_detect_param
                )
        if is_unseen_task:
            utd_saved_url = self.get_parameters('UTD_SAVED_URL')
            fd, name = tempfile.mkstemp()
            joblib.dump(name, tasks)
            os.close(fd)
            out_file = FileOps.join_path(utd_saved_url, FileOps.get_file_hash(name))

            FileOps.upload(name, out_file)
            os.remove(name)

        # self._report_task_info(None, K8sResourceKindStatus.COMPLETED.value, res, kind="inference")
        return res, is_unseen_task

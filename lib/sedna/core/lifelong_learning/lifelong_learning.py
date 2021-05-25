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
from sedna.core.base import JobBase
from sedna.common.file_ops import FileOps
from sedna.common.constant import K8sResourceKind, K8sResourceKindStatus
from sedna.common.config import BaseConfig, Context
from sedna.common.log import sednaLogger
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.multi_task_learning import MulTaskLearning
from sedna.algorithms.multi_task_learning.task_jobs.artifact import TaskGroup
from sedna.service.client import KBClient


class LifelongLearning(JobBase):
    """
    Lifelong learning Experiment
    """

    def __init__(self, estimator, method_selection=None):
        estimator = MulTaskLearning(
            estimator=estimator,
            method_selection=method_selection,
        )
        config = dict(
            ll_kb_server=Context.get_parameters("KB_SERVER"),
            output_url=Context.get_parameters("OUTPUT_URL")
        )
        task_index = FileOps.join_path(config['output_url'], 'deploy', 'index.pkl')
        config['task_index'] = task_index
        super(LifelongLearning, self).__init__(estimator=estimator, config=config)
        self.job_kind = K8sResourceKind.LIFELONG_JOB.value
        self.kb_server = KBClient(kbserver=self.config.ll_kb_server)

    def _update_db(self, task_info: TaskGroup):
        _id = self.kb_server.update_db(task_info)

        if not _id:
            self.log.error(f"KB update Fail !")
            return
        # res = self.kb_server.check_job_status(_id)
        return

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

        task_info = self.estimator.estimator.task_groups
        FileOps.upload(self.estimator.estimator.task_index_url, self.config.task_index)
        self._update_db(task_info)

        self._report_task_info(None, K8sResourceKindStatus.COMPLETED.value, res)
        sednaLogger.info("Lifelong learning Experiment Finished")
        return callback_func(self.estimator, res) if callback_func else res

    def update(self, train_data, valid_data=None, post_process=None, **kwargs):
        return self.train(
            train_data=train_data,
            valid_data=valid_data,
            post_process=post_process,
            action="update",
            **kwargs
        )

    def inference(self, data=None, post_process=None, **kwargs):
        res, tasks = self.estimator.predict(
            data=data, post_process=post_process, **kwargs
        )
        utd = self.get_parameters("UTD_NAME", "TaskAttr")
        utd_parameters = self.get_parameters("UTD_PARAMETERS", {})
        is_unseen_task = False
        if utd:
            try:
                unseen_task_detect_algorithm = ClassFactory.get_cls(ClassType.UTD, utd)()
            except ValueError as err:
                sednaLogger.error("Lifelong learning Experiment Inference [UTD] : {}".format(err))
            else:
                is_unseen_task = unseen_task_detect_algorithm(tasks=tasks, result=res, **utd_parameters)
        self._report_task_info(None, K8sResourceKindStatus.COMPLETED.value, res, kind="inference")
        return res, is_unseen_task

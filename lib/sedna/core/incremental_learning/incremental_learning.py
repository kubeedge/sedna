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
from copy import deepcopy
from sedna.common.log import sednaLogger
from sedna.common.file_ops import FileOps
from sedna.common.constant import K8sResourceKind, K8sResourceKindStatus
from sedna.common.class_factory import ClassFactory, ClassType
from ..base import JobBase

__all__ = ("IncrementalLearning",)


class IncrementalLearning(JobBase):
    """
    Incremental learning Experiment
    """

    def __init__(self, estimator, config=None):
        super(IncrementalLearning, self).__init__(estimator=estimator, config=config)
        self.job_kind = K8sResourceKind.INCREMENTAL_JOB.value
        FileOps.clean_folder([self.config.model_url], clean=False)
        self.log.info("Incremental learning Experiment model prepared")

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(ClassType.CALLBACK, post_process)

        self.log.info("Incremental learning Experiment Train Start")
        res = self.estimator.train(train_data=train_data, valid_data=valid_data, **kwargs)
        self.estimator.save(self.model_path)
        self._report_task_info(None, K8sResourceKindStatus.COMPLETED.value, res)
        sednaLogger.info("Incremental learning Experiment Finished")
        return callback_func(self.estimator) if callback_func else self.estimator

    def inference(self, data=None, post_process=None, **kwargs):

        if not self.estimator.has_load:
            self.estimator.load(self.model_path)
        hem = self.get_parameters("HEM_NAME")
        hem_parameters = self.get_parameters("HEM_PARAMETERS")
        if hem is None:
            hem = self.config.get("hem_name") or "IBT"
        if hem_parameters is None:
            hem_parameters = {}
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(ClassType.CALLBACK, post_process)
        res = self.estimator.predict(data, **kwargs)
        rsl = callback_func(deepcopy(res)) if callback_func else res
        is_hard_example = False
        try:
            hard_example_mining_algorithm = ClassFactory.get_cls(ClassType.HEM, hem)(**hem_parameters)
        except ValueError as err:
            sednaLogger.error("Incremental learning Experiment Inference [HEM] : {}".format(err))
        else:
            is_hard_example = hard_example_mining_algorithm(rsl)
        return res, rsl, is_hard_example

    def evaluate(self, data, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process:
            callback_func = ClassFactory.get_cls(ClassType.CALLBACK, post_process)
        res = self.estimator.evaluate(data=data, **kwargs)
        if callback_func:
            res = callback_func(res)
        self._report_task_info(None, K8sResourceKindStatus.COMPLETED.value, res, kind="eval")

        return res


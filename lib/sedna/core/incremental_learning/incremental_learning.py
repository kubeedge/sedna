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

import json
import os
from copy import deepcopy
from sedna.common.file_ops import FileOps
from sedna.common.constant import K8sResourceKind, K8sResourceKindStatus
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.core.base import JobBase

__all__ = ("IncrementalLearning",)


class IncrementalLearning(JobBase):
    """
    Incremental learning Experiment
    """

    def __init__(self, estimator, config=None):
        super(IncrementalLearning, self).__init__(
            estimator=estimator, config=config)

        self.model_urls = self.get_parameters(
            "MODEL_URLS")  # use in evaluation
        self.job_kind = K8sResourceKind.INCREMENTAL_JOB.value
        FileOps.clean_folder([self.config.model_url], clean=False)
        hem = self.get_parameters("HEM_NAME")
        hem_parameters = self.get_parameters("HEM_PARAMETERS")

        try:
            hem_parameters = json.loads(hem_parameters)
            if isinstance(hem_parameters, (list, tuple)):
                if isinstance(hem_parameters[0], dict):
                    hem_parameters = {
                        p["key"]: p.get("value", "")
                        for p in hem_parameters if "key" in p
                    }
                else:
                    hem_parameters = dict(hem_parameters)
        except Exception:
            hem_parameters = None

        if hem is None:
            hem = self.config.get("hem_name") or "IBT"

        if hem_parameters is None:
            hem_parameters = {}
        self.hard_example_mining_algorithm = ClassFactory.get_cls(
            ClassType.HEM, hem)(**hem_parameters)

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        res = self.estimator.train(
            train_data=train_data, valid_data=valid_data, **kwargs)
        model_paths = self.estimator.save(self.model_path)
        task_info_res = self.estimator.model_info(
            model_paths, result=res, relpath=self.config.data_path_prefix)
        self.report_task_info(
            None, K8sResourceKindStatus.COMPLETED.value, task_info_res)
        return callback_func(
            self.estimator) if callback_func else self.estimator

    def inference(self, data=None, post_process=None, **kwargs):
        if not self.estimator.has_load:
            self.estimator.load(self.model_path)

        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        res = self.estimator.predict(data, **kwargs)
        rsl = callback_func(deepcopy(res)) if callback_func else res
        is_hard_example = False

        if self.hard_example_mining_algorithm:
            is_hard_example = self.hard_example_mining_algorithm(rsl)
        return res, rsl, is_hard_example

    def evaluate(self, data, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        final_res = []
        if self.model_urls:
            for model_url in self.model_urls.split(";"):
                if not model_url.strip():
                    continue
                self.estimator.model_save_path = model_url
                res = self.estimator.evaluate(
                    data=data, model_path=model_url, **kwargs)
                if callback_func:
                    res = callback_func(res)
                self.log.info(f"Evaluation with {model_url} : {res} ")
                task_info_res = self.estimator.model_info(
                    model_url, result=res,
                    relpath=self.config.data_path_prefix)
                if isinstance(task_info_res, (list, tuple)
                              ) and len(task_info_res):
                    task_info_res = list(task_info_res)[0]
                final_res.append(task_info_res)
        else:
            model_url = self.config.model_url
            res = self.estimator.evaluate(data=data, **kwargs)
            if callback_func:
                res = callback_func(res)
            self.log.info(f"Evaluation with {model_url} : {res} ")
            final_res = self.estimator.model_info(
                model_url, result=res, relpath=self.config.data_path_prefix)
        self.report_task_info(None, K8sResourceKindStatus.COMPLETED.value,
                              final_res, kind="eval")

        return final_res

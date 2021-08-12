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

import os.path
import json

from sedna.common.log import LOGGER
from sedna.common.file_ops import FileOps
from sedna.common.config import BaseConfig
from sedna.common.config import Context
from sedna.common.constant import K8sResourceKind
from sedna.service.client import LCClient
from sedna.backend import set_backend
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('JobBase',)


class JobBase:
    """ sedna feature base class """
    parameters = Context

    def __init__(self, estimator, config=None):
        super(JobBase, self).__init__()
        self.config = BaseConfig()
        if config:
            self.config.from_json(config)
        self.log = LOGGER
        self.estimator = set_backend(estimator=estimator, config=self.config)
        self.job_kind = K8sResourceKind.DEFAULT.value
        self.job_name = self.config.job_name or self.config.service_name
        self.worker_name = self.config.worker_name or self.job_name

    @property
    def model_path(self):
        if os.path.isfile(self.config.model_url):
            return self.config.model_url
        return self.get_parameters('model_path') or FileOps.join_path(
            self.config.model_url, self.estimator.model_name)

    def train(self, **kwargs):
        raise NotImplementedError

    def inference(self, x=None, post_process=None, **kwargs):

        res = self.estimator.predict(x, kwargs=kwargs)
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        return callback_func(res) if callback_func else res

    def evaluate(self, data, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        res = self.estimator.evaluate(data=data, **kwargs)
        return callback_func(res) if callback_func else res

    def get_parameters(self, param, default=None):
        return self.parameters.get_parameters(param=param, default=default)

    def report_task_info(self, task_info, status, results, kind="train"):
        message = {
            "name": self.worker_name,
            "namespace": self.config.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": kind,
            "status": status,
            "results": results
        }
        if task_info:
            message["ownerInfo"] = task_info
        try:
            LCClient.send(self.config.lc_server, self.worker_name, message)
        except Exception as err:
            self.log.error(err)

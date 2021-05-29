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
import os.path

from sedna.common.log import sednaLogger
from sedna.common.file_ops import FileOps
from sedna.common.config import BaseConfig
from sedna.common.config import Context
from sedna.common.constant import K8sResourceKind
from sedna.service.client import LCClient
from sedna.backend import set_backend
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('JobBase',)


class DistributedWorker:
    """"Class of Distributed Worker use to manage all jobs"""
    # original params
    __worker_path__ = None
    __worker_module__ = None
    # id params
    __worker_id__ = 0

    def __init__(self):
        DistributedWorker.__worker_id__ += 1
        self._worker_id = DistributedWorker.__worker_id__
        self.timeout = 0

    @property
    def worker_id(self):
        """Property: worker_id."""
        return self._worker_id

    @worker_id.setter
    def worker_id(self, value):
        """Setter: set worker_id with value.

        :param value: worker id
        :type value: int
        """
        self._worker_id = value


class JobBase(DistributedWorker):
    """ sedna feature base class """
    parameters = Context

    def __init__(self, estimator, config=None):
        super(JobBase, self).__init__()
        self.config = BaseConfig()
        if config:
            self.config.from_json(config)
        self.log = sednaLogger
        self.estimator = set_backend(estimator=estimator, config=self.config)
        self.job_kind = K8sResourceKind.DEFAULT.value
        self.job_name = self.config.job_name or self.config.service_name
        self.worker_name = self.config.worker_name or f"{self.job_name}-{self.worker_id}"

    @property
    def model_path(self):
        return self.get_parameters('model_path') or FileOps.join_path(self.config.model_url, self.estimator.model_name)

    def train(self, **kwargs):
        pass

    def inference(self, x=None, post_process=None, **kwargs):

        res = self.estimator.predict(x, kwargs=kwargs)
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(ClassType.CALLBACK, post_process)
        return callback_func(res) if callback_func else res

    def evaluate(self, data, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(ClassType.CALLBACK, post_process)
        res = self.estimator.evaluate(data=data, **kwargs)
        return callback_func(res) if callback_func else res

    def get_parameters(self, param, default=None):
        return self.parameters.get_parameters(param=param, default=default)

    def _report_task_info(self, task_info, status, metrics, kind="train", model=None):
        if model:
            _, _type = os.path.splitext(model)
            _url = FileOps.remove_path_prefix(model,
                                              self.config.data_path_prefix)
            results = [{
                "format": _type.lstrip("."),
                "url": _url,
                "metrics": metrics
            }]
            if _type == ".pb":
                _url = FileOps.remove_path_prefix(os.path.dirname(model),
                                                  self.config.data_path_prefix)
                results.append(
                    {
                        "format": "ckpt",
                        "url": _url,
                        "metrics": metrics
                    }
                )
        else:
            ckpt_model_url = FileOps.remove_path_prefix(self.config.model_url,
                                                        self.config.data_path_prefix)
            pb_model_url = FileOps.remove_path_prefix(self.model_path,
                                                      self.config.data_path_prefix)

            ckpt_result = {
                "format": "ckpt",
                "url": ckpt_model_url,
                "metrics": metrics
            }

            pb_result = {
                "format": "pb",
                "url": pb_model_url,
                "metrics": metrics
            }

            results = [ckpt_result, pb_result]

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
            with open(FileOps.join_path(self.config.model_url, "lc_msg.json"), 'w') as f_out:
                json.dump(message, f_out)

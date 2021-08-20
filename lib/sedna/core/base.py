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
import threading
import gc
import time
import json

from sedna.common.log import LOGGER
from sedna.common.file_ops import FileOps
from sedna.common.config import BaseConfig
from sedna.common.config import Context
from sedna.common.constant import K8sResourceKind
from sedna.common.constant import K8sResourceKindStatus
from sedna.service.client import LCClient
from sedna.backend import set_backend
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('JobBase',)


class ModelLoadingThread(threading.Thread):
    """ MulThread support hot model loading"""
    MODEL_MANIPULATION_SEM = threading.Semaphore(1)
    MODEL_CHECK_TIME = max(
        int(Context.get_parameters("MODEL_POLL_PERIOD_SECONDS", "60")), 1
    )
    MODEL_HOT_UPDATE_CONF = Context.get_parameters("MODEL_HOT_UPDATE_CONFIG")
    MODEL_TEMP_SAVE = Context.get_parameters("MODEL_TEMP", "/tmp")

    def __init__(self,
                 estimator,
                 job_kind,
                 job_name,
                 worker_name,
                 namespace,
                 lc_server=None,
                 version="latest"
                 ):
        self.production_estimator = estimator
        self.job_name = job_name
        self.job_kind = job_kind
        self.namespace = namespace
        self.worker_name = worker_name
        self.lc_server = lc_server
        self.version = version
        self.temp_path = (self.MODEL_TEMP_SAVE
                          if os.path.isdir(self.MODEL_TEMP_SAVE)
                          else os.path.dirname(self.MODEL_TEMP_SAVE))

        super(ModelLoadingThread, self).__init__()

    def report_task_info(self, status, kind="deploy"):
        if not self.lc_server:
            return
        message = {
            "name": self.worker_name,
            "namespace": self.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": kind,
            "status": status
        }
        try:
            LCClient.send(self.lc_server, self.worker_name, message)
        except Exception as err:
            LOGGER.error(err)

    def run(self):
        conf = FileOps.join_path(self.temp_path, "tmp.conf")
        while 1:
            time.sleep(self.MODEL_CHECK_TIME)
            if not self.MODEL_HOT_UPDATE_CONF:
                continue
            conf = FileOps.download(self.MODEL_HOT_UPDATE_CONF, conf)
            if not FileOps.exists(conf):
                continue
            with open(conf, "r") as fin:
                try:
                    conf_msg = json.load(fin)
                    model_msg = conf_msg["model_config"]
                    latest_version = str(model_msg["model_update_time"])
                    model = FileOps.download(
                        model_msg["model_path"],
                        FileOps.join_path(
                            self.temp_path, f"model.{latest_version}"
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    LOGGER.error(f"fail to parse model hot update config: "
                                 f"{self.MODEL_HOT_UPDATE_CONF}")
                    continue
            if not (model and FileOps.exists(model)):
                continue
            if latest_version == self.version:
                continue
            self.version = latest_version
            with self.MODEL_MANIPULATION_SEM:
                LOGGER.info(f"Update model start with version {self.version}")
                try:
                    self.production_estimator.load(model)
                    status = K8sResourceKindStatus.COMPLETED.value
                    LOGGER.info(f"Update model complete "
                                f"with version {self.version}")
                except Exception as e:
                    LOGGER.error(f"fail to update model: {e}")
                    status = K8sResourceKindStatus.FAILED.value
                self.report_task_info(status=status)
            gc.collect()


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
        self.namespace = self.config.namespace or self.job_name
        self.lc_server = self.config.lc_server
        if str(
                self.get_parameters("MODEL_HOT_UPDATE", "False")
        ).lower() == "true":
            ModelLoadingThread(
                self.estimator,
                self.job_kind,
                self.job_name,
                self.worker_name,
                self.namespace,
                self.lc_server
            ).start()

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
            "namespace": self.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": kind,
            "status": status,
            "results": results
        }
        if task_info:
            message["ownerInfo"] = task_info
        try:
            LCClient.send(self.lc_server, self.worker_name, message)
        except Exception as err:
            self.log.error(err)

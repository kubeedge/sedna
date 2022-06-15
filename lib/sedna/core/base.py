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
import tempfile

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
    """Hot model loading with multithread support"""
    MODEL_MANIPULATION_SEM = threading.Semaphore(1)

    def __init__(self,
                 estimator,
                 callback=None,
                 version="latest"
                 ):
        self.run_flag = True
        hot_update_conf = Context.get_parameters("MODEL_HOT_UPDATE_CONFIG")
        if not hot_update_conf:
            LOGGER.error("As `MODEL_HOT_UPDATE_CONF` unset a value, skipped")
            self.run_flag = False
        model_check_time = int(Context.get_parameters(
            "MODEL_POLL_PERIOD_SECONDS", "60")
        )
        if model_check_time < 1:
            LOGGER.warning("Catch an abnormal value in "
                           "`MODEL_POLL_PERIOD_SECONDS`, fallback with 60")
            model_check_time = 60
        self.hot_update_conf = hot_update_conf
        self.check_time = model_check_time
        self.production_estimator = estimator
        self.callback = callback
        self.version = version
        self.temp_path = tempfile.gettempdir()
        super(ModelLoadingThread, self).__init__()

    def run(self):
        while self.run_flag:
            time.sleep(self.check_time)
            conf = FileOps.download(self.hot_update_conf)
            if not (conf and FileOps.exists(conf)):
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
                                 f"{self.hot_update_conf}")
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
                if self.callback:
                    self.callback(
                        task_info=None, status=status, kind="deploy"
                    )
            gc.collect()


class JobBase:
    """ sedna feature base class """
    parameters = Context

    def __init__(self, estimator, config=None):
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
                self.report_task_info
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

    def report_task_info(self, task_info, status, results=None, kind="train"):
        message = {
            "name": self.worker_name,
            "namespace": self.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": kind,
            "status": status
        }
        if results:
            message["results"] = results
        if task_info:
            message["ownerInfo"] = task_info
        try:
            LCClient.send(self.lc_server, self.worker_name, message)
        except Exception as err:
            self.log.error(err)

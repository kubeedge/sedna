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

from copy import deepcopy
from sedna.common.utils import get_host_ip
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.service.server import InferenceServer
from sedna.service.client import ModelClient, LCReporter
from sedna.common.constant import K8sResourceKind

from sedna.core.base import JobBase

__all__ = ("JointInference", "TSBigModelService")


class TSBigModelService(JobBase):
    """
    Large model services implemented
    Provides RESTful interfaces for large-model inference.
    """

    def __init__(self, estimator=None, config=None):
        super(TSBigModelService, self).__init__(
            estimator=estimator, config=config)
        self.local_ip = self.get_parameters("BIG_MODEL_BIND_IP", get_host_ip())
        self.port = int(self.get_parameters("BIG_MODEL_BIND_PORT", "5000"))

    def start(self):
        if callable(self.estimator):
            self.estimator = self.estimator()
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"{self.model_path} miss")
        else:
            self.estimator.load(self.model_path)
        app_server = InferenceServer(model=self, servername=self.job_name,
                                     host=self.local_ip, http_port=self.port)
        app_server.start()

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        self.log.warn("Joint inference Experiment should offer model file")
        invert_train = getattr(self, "train", None)
        if not callable(invert_train):
            raise RuntimeError("NO SUPPORT METHOD")

        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        self.log.info("Joint inference Experiment Train Start")
        _ = self.estimator.train(
            train_data=train_data, valid_data=valid_data, **kwargs)
        self.estimator.save(self.model_path)
        self.log.info("Joint inference Experiment Train Finished")
        return callback_func(
            self.estimator) if callback_func else self.estimator

    def inference(self, data=None, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        res = self.estimator.predict(data, **kwargs)

        if callback_func:
            res = callback_func(res)
        return res


class JointInference(JobBase):
    """
   Joint inference Experiment
   """

    def __init__(self, estimator=None, config=None):
        super(JointInference, self).__init__(
            estimator=estimator, config=config)
        self.job_kind = K8sResourceKind.JOINT_INFERENCE_SERVICE.value
        self.local_ip = get_host_ip()
        self.remote_ip = self.get_parameters(
            "BIG_MODEL_BIND_IP", self.local_ip)
        self.port = int(self.get_parameters("BIG_MODEL_BIND_PORT", "5000"))

        report_msg = {
            "name": self.worker_name,
            "namespace": self.config.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": "inference",
            "results": []
        }
        period_interval = int(self.get_parameters("LC_PERIOD", "30"))
        self.lc_reporter = LCReporter(message=report_msg,
                                      period_interval=period_interval)
        self.lc_reporter.setDaemon(True)
        self.lc_reporter.start()

        if callable(self.estimator):
            self.estimator = self.estimator()
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"{self.model_path} miss")
        else:
            self.estimator.load(self.model_path)
        self.cloud = ModelClient(service_name=self.job_name,
                                 host=self.remote_ip, port=self.port)

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        self.log.warn("Joint inference Experiment should offer model file")
        invert_train = getattr(self.estimator, "train", None)
        if not callable(invert_train):
            raise RuntimeError("NO SUPPORT METHOD")

        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        self.log.info("Joint inference Experiment Train Start")
        _ = self.estimator.train(
            train_data=train_data, valid_data=valid_data, **kwargs)
        self.estimator.save(self.model_path)
        self.log.info("Joint inference Experiment Train Finished")
        return callback_func(
            self.estimator) if callback_func else self.estimator

    def inference(self, data=None, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        self.log.info("Joint Inference Experiment edge Inference Start")
        res = self.estimator.predict(data, **kwargs)
        self.log.info("Joint Inference Experiment edge Inference Done")
        edge_result = deepcopy(res)

        if callback_func:
            res = callback_func(res)

        self.lc_reporter.update_for_edge_inference()
        hem = self.get_parameters("HEM_NAME")
        hem_parameters = self.get_parameters("HEM_PARAMETERS")
        if hem is None:
            hem = self.config.get("hem_name") or "IBT"
        if hem_parameters is None:
            hem_parameters = {}

        is_hard_example = False
        cloud_result = None

        try:
            hard_example_mining_algorithm = ClassFactory.get_cls(
                ClassType.HEM, hem)()
        except ValueError as err:
            self.log.error(
                "Joint Inference Experiment Inference [HEM] : {}".format(err))
        else:
            is_hard_example = hard_example_mining_algorithm(
                res, **hem_parameters)
            if is_hard_example:
                cloud_result = self.cloud.inference(
                    data.tolist(), post_process=post_process, **kwargs)
                self.lc_reporter.update_for_collaboration_inference()
        return [is_hard_example, res, edge_result, cloud_result]

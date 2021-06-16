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
import json
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

    def __init__(self, estimator=None):
        """
        Initial a big model service for JointInference
        :param estimator: Customize estimator
        """

        super(TSBigModelService, self).__init__(estimator=estimator)
        self.local_ip = self.get_parameters("BIG_MODEL_BIND_IP", get_host_ip())
        self.port = int(self.get_parameters("BIG_MODEL_BIND_PORT", "5000"))

    def start(self):
        """
        Start inference rest server
        :return:
        """

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
        """todo: no support yet"""

    def inference(self, data=None, post_process=None, **kwargs):
        """
        Inference task for IncrementalLearning
        :param data: inference sample
        :param post_process: post process
        :param kwargs: params for inference of big model
        :return: inference result
        """

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
   Joint inference
   """

    def __init__(self, estimator=None):
        """
        Initial a JointInference Job
        :param estimator: Customize estimator
        """

        super(JointInference, self).__init__(estimator=estimator)
        self.job_kind = K8sResourceKind.JOINT_INFERENCE_SERVICE.value
        self.local_ip = get_host_ip()
        self.remote_ip = self.get_parameters(
            "BIG_MODEL_IP", self.local_ip)
        self.port = int(self.get_parameters("BIG_MODEL_PORT", "5000"))

        report_msg = {
            "name": self.worker_name,
            "namespace": self.config.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": "inference",
            "results": []
        }
        period_interval = int(self.get_parameters("LC_PERIOD", "30"))
        self.lc_reporter = LCReporter(lc_server=self.config.lc_server,
                                      message=report_msg,
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
        self.hard_example_mining_algorithm = self.initial_hem

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        """todo: no support yet"""

    def inference(self, data=None, post_process=None, **kwargs):
        """
        Inference task for IncrementalLearning
        :param data: inference sample
        :param post_process: post process
        :param kwargs: params for inference of customize estimator
        :return: if is hard sample, real result, little model result, big model result
        """

        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        res = self.estimator.predict(data, **kwargs)
        edge_result = deepcopy(res)

        if callback_func:
            res = callback_func(res)

        self.lc_reporter.update_for_edge_inference()

        is_hard_example = False
        cloud_result = None

        if self.hard_example_mining_algorithm:
            is_hard_example = self.hard_example_mining_algorithm(res)
            if is_hard_example:
                cloud_result = self.cloud.inference(
                    data.tolist(), post_process=post_process, **kwargs)
                self.lc_reporter.update_for_collaboration_inference()
        return [is_hard_example, res, edge_result, cloud_result]

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
import sys

from sedna.common.utils import get_host_ip, flatten_nested_list
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.service.server import InferenceServer
from sedna.service.client import ModelClient, LCReporter
from sedna.common.constant import K8sResourceKind
from sedna.core.base import JobBase
from sedna.common.benchmark import FTimer

__all__ = ("EdgeInference", "CloudInference")


class CloudInference(JobBase):
    """
    Cloud model services implemented
    Provides RESTful interfaces for cloud-model inference.
    """

    def __init__(self, estimator=None, config=None):
        super(CloudInference, self).__init__(
            estimator=estimator, config=config)
        self.local_ip = self.get_parameters("CLOUD_MODEL_BIND_IP", get_host_ip())
        self.port = int(self.get_parameters("CLOUD_MODEL_BIND_PORT", "5000"))

    def start(self):
        if callable(self.estimator):
            self.estimator = self.estimator()
        # if not os.path.exists(self.model_path):
        #     raise FileExistsError(f"{self.model_path} miss")
        # else:
        self.estimator.load()
        app_server = InferenceServer(model=self, servername=self.job_name,
                                     host=self.local_ip, http_port=self.port)
        app_server.start()

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        """todo: no support yet"""

    def inference(self, data=None, post_process=None, **kwargs):
        self.log.info(f"in cloud inference")
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        with FTimer(f"{self.worker_name}_cloud_inference"):
            self.log.info(f"calling cloud predict")
            res = self.estimator.predict(data, **kwargs)

        if callback_func:
            res = callback_func(res)

        return res


class EdgeInference(JobBase):
    """
   Edge Inference
   """

    def __init__(self, estimator=None, config=None):
        super(EdgeInference, self).__init__(
            estimator=estimator, config=config)
        self.job_kind = K8sResourceKind.DNN_PARTITIONING_SERVICE.value
        self.local_ip = get_host_ip()
        self.remote_ip = self.get_parameters(
            "CLOUD_MODEL_IP", self.local_ip)
        self.port = int(self.get_parameters("CLOUD_MODEL_PORT", "5000"))

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
        # self.lc_reporter.start()

        if callable(self.estimator):
            self.estimator = self.estimator()
        # if not os.path.exists(self.model_path):
        #     raise FileExistsError(f"{self.model_path} miss")
        # else:
        #     # We are using a PyTorch model which requires explicit weights loading.
        #     self.log.info("Estimator -> Loading model and weights")

        # We should pass the model path but, because it's in the container logic, we don't pass anything.
        self.estimator.load()
        self.log.info(f"cloud host local ip: {self.local_ip}")
        self.log.info(f"cloud host: {self.remote_ip}")
        self.log.info(f"cloud port: {self.port}")

        self.cloud = ModelClient(service_name=self.job_name,
                                 host=self.remote_ip, port=self.port)
        
        self.log.info(f"cloud obj: {self.cloud}")
        # self.hard_example_mining_algorithm = self.initial_hem

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        """todo: no support yet"""

    def inference(self, data=None, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        with FTimer(f"{os.uname()[1]}_edge_inference"):
            res = self.estimator.predict(data, **kwargs)
        edge_result = deepcopy(res)

        self.log.info(f"edge inference completed")

        if callback_func:
            res = callback_func(res)

        # self.lc_reporter.update_for_edge_inference()

        # is_hard_example = False
        cloud_result = None

         # Send detection+tracking results to cloud
        # edge_result

        #if edge_result != None:
        with FTimer(f"{os.uname()[1]}_cloud_inference_and_transmission"):
            self.log.info(f"calling cloud inference")
            cloud_result = self.cloud.inference(edge_result, post_process=post_process, **kwargs)
            self.log.info(f"cloud inference completed")

        return [None, cloud_result, edge_result, None]

        # if self.hard_example_mining_algorithm:
        #     is_hard_example = self.hard_example_mining_algorithm(res)
        #     if is_hard_example:
        #         with FTimer(f"{os.uname()[1]}_cloud_inference_and_transmission"):
        #             cloud_result = self.cloud.inference(
        #                 data.tolist(), post_process=post_process, **kwargs)

        #         size = sys.getsizeof(0) * len(flatten_nested_list(cloud_result['result']))
        #         self.log.info(f"Received data: {size} bytes.")
        #         self.lc_reporter.update_for_collaboration_inference()
        # return [is_hard_example, res, edge_result, cloud_result]

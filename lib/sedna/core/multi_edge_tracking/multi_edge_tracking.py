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
from sedna.service.server import ReIDServer
from sedna.service.reid_endpoint import ReID
from sedna.service.client import LCReporter
from sedna.common.constant import K8sResourceKind
from sedna.core.base import JobBase
from sedna.common.benchmark import FTimer
from sedna.common.log import LOGGER

__all__ = ("MultiObjectTracking", "ReIDService", "ObjectDetector")


class ReIDService(JobBase):
    """
    Re-Identification model services
    Provides RESTful interfaces to execute the object ReIdentification.
    """

    def __init__(self, estimator=None, config=None):
        super(ReIDService, self).__init__(
            estimator=estimator, config=config)
        self.log.info("Starting ReID service")

        self.local_ip = self.get_parameters("REID_MODEL_BIND_IP", get_host_ip())
        self.port = int(self.get_parameters("REID_MODEL_BIND_PORT", "5000"))

    def start(self):
        if callable(self.estimator):
            self.estimator = self.estimator()
        # The cloud instance only runs a distance function to do the ReID
        # We don't load any model at this stage
        # if not os.path.exists(self.model_path):
        #     raise FileExistsError(f"{self.model_path} miss")
        # else:
        #     # self.estimator.load(self.model_path)
        app_server = ReIDServer(model=self, servername=self.job_name,
                                     host=self.local_ip, http_port=self.port)
        app_server.start()

    def inference(self, data=None, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        with FTimer(f"{self.worker_name}_reid"):
            res = self.estimator.predict(data, **kwargs)

        if callback_func:
            res = callback_func(res)

        return res


class MultiObjectTracking(JobBase):
    """
   MultiObject Tracking service.
   """

    def __init__(self, estimator=None, config=None):
        super(MultiObjectTracking, self).__init__(
            estimator=estimator, config=config)
        self.log.info("Loading MultiObjectTracking module")

        self.job_kind = K8sResourceKind.MULTI_EDGE_TRACKING_SERVICE.value
        self.local_ip = get_host_ip()
        self.remote_ip = self.get_parameters(
            "REID_MODEL_BIND_IP", self.local_ip)
        self.port = int(self.get_parameters("REID_MODEL_PORT", "5000"))

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

        if estimator is None:
            self.log.error("ERROR! Estimator is not set!")

        if callable(self.estimator):
            self.estimator = self.estimator()
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"{self.model_path} miss")
        else:
            # We are using a PyTorch model which requires explicit weights loading.
            self.log.info("Estimator -> Loading model and weights")
            self.estimator.load(self.model_path)
            #self.estimator.load_weights()

            self.log.info("Estimator -> Evaluating model ..")
            self.estimator.evaluate()

        self.cloud = ReID(service_name=self.job_name,
                                 host=self.remote_ip, port=self.port)

    def inference(self, data=None, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        with FTimer(f"{os.uname()[1]}_mot"):
            res = self.estimator.predict(data, **kwargs)
        edge_result = deepcopy(res)

        if callback_func:
            res = callback_func(res)

        self.lc_reporter.update_for_edge_inference()
        # Send detection+tracking results to cloud
        # edge_result
        
        if edge_result != None:
            with FTimer(f"upload_plus_reid"):
                cres = self.cloud.reid(edge_result, post_process=post_process, **kwargs)

        return [None, cres, edge_result, None]


class ObjectDetector(JobBase):
    """
   ObjectDetector service.
   """

    def __init__(self, estimator=None, config=None):
        super(ObjectDetector, self).__init__(
            estimator=estimator, config=config)
        self.log.info("Loading ObjectDetector module")
        self.job_kind = K8sResourceKind.MULTI_EDGE_TRACKING_SERVICE.value

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

        if estimator is None:
            self.log.error("ERROR! Estimator is not set!")

        if callable(self.estimator):
            self.estimator = self.estimator()
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"{self.model_path} miss")
        else:
            # We are using a PyTorch model which requires explicit weights loading.
            self.log.info("Estimator -> Loading model and weights")
            self.estimator.load(self.model_path)

            self.log.info("Estimator -> Evaluating model ..")
            self.estimator.evaluate()

    def inference(self, data=None, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        with FTimer(f"{os.uname()[1]}_object_detection"):
             detection_result = self.estimator.predict(data, **kwargs)

        if callback_func:
            detection_result = callback_func(detection_result)

        self.lc_reporter.update_for_edge_inference()

        #if edge_result != None:
        #    with FTimer(f"upload_plus_reid"):
        #        cres = self.cloud.reid(edge_result, post_process=post_process, **kwargs)

        return detection_result


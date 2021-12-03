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

import base64
import io, json
import os, queue
from copy import deepcopy
import threading
import time
from PIL import Image

import numpy as np
import requests
from sedna.common.log import LOGGER

from sedna.common.utils import get_host_ip
from sedna.common.class_factory import ClassFactory, ClassType

from sedna.service.fe_endpoint import FE
from sedna.service.kafka_manager import KafkaConsumerThread, KafkaProducer
from sedna.service.reid_endpoint import ReID

from sedna.core.base import JobBase
from sedna.common.benchmark import FTimer

from sedna.service.server import FEServer
from sedna.service.server import ReIDServer

import distutils.core

__all__ = ("FEService", "ReIDService", "ObjectDetector")

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
        self.kafka_enabled = bool(distutils.util.strtobool(self.get_parameters("KAFKA_ENABLED", "False")))

        if self.kafka_enabled:
            self.log.debug("Kafka support enabled in YAML file")
            self.kafka_address = self.get_parameters("KAFKA_BIND_IPS", ["7.182.9.110"])
            self.kafka_port = self.get_parameters("KAFKA_BIND_PORTS", [32669])

            if isinstance(self.kafka_address, str):
                self.log.debug(f"Parsing string received from K8s controller {self.kafka_address},{self.kafka_port}")
                self.kafka_address = self.kafka_address.split("|")
                self.kafka_port = self.kafka_port.split("|")
            
            self.sync_queue = queue.Queue()

            self.producer = KafkaProducer(self.kafka_address, self.kafka_port, topic=["reid"])
            self.consumer = KafkaConsumerThread(self.kafka_address, self.kafka_port, topic=["feature_extraction"], sync_queue=self.sync_queue)

        # Operating Mode
        self.op_mode = "detection"
        self.target = None

        
    def start(self):
        if callable(self.estimator):
            self.estimator = self.estimator()

        StatusSyncThread(self.update_operational_mode)

        # The cloud instance only runs a distance function to do the ReID
        # We don't load any model here.
        if self.kafka_enabled:
            self.log.debug("Creating sync_inference thread")
            self.fetch_data()
        else:
            self.log.debug("Starting default REST webservice")
            app_server = ReIDServer(model=self, servername=self.job_name, host=self.local_ip, http_port=self.port)
            app_server.start()

    def fetch_data(self):
        while True:
            token = self.sync_queue.get()
            self.log.debug(f'Data consumed')
            try:
                self.inference(token)
            except Exception as e:
                self.log.debug(f"Error processing received data: {e}")

            self.sync_queue.task_done()

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

    def update_operational_mode(self, status):
        self.log.debug("Configuration update triggered")

        if status == None:
            return

        self.op_mode = status['op_mode']


class FEService(JobBase):
    """
   Feature Extraction service.
   """

    def __init__(self, estimator=None, config=None):
        super(FEService, self).__init__(
            estimator=estimator, config=config)
        self.log.info("Loading Feature Extraction module")

        # Port and IP of the service this pod will host (local)
        self.local_ip = self.get_parameters(get_host_ip())
        self.local_port = int(self.get_parameters("FE_MODEL_BIND_PORT", "6000"))

        # Port and IP of the service this pod will contact (remote)
        self.remote_ip = self.get_parameters("REID_MODEL_BIND_URL", self.local_ip)
        self.remote_port = int(self.get_parameters("REID_MODEL_PORT", "5000"))
        self.kafka_enabled = bool(distutils.util.strtobool(self.get_parameters("KAFKA_ENABLED", "False")))
        self.sync_queue = queue.Queue()

        if self.kafka_enabled:
            self.log.debug("Kafka support enabled in YAML file")
            self.kafka_address = self.get_parameters("KAFKA_BIND_IPS", ["7.182.9.110"])
            self.kafka_port = self.get_parameters("KAFKA_BIND_PORTS", [32669])
            
            if isinstance(self.kafka_address, str):
                self.log.debug(f"Parsing string received from K8s controller {self.kafka_address},{self.kafka_port}")
                self.kafka_address = self.kafka_address.split("|")
                self.kafka_port = self.kafka_port.split("|")

            self.producer = KafkaProducer(self.kafka_address, self.kafka_port, topic=["feature_extraction"])
            self.consumer = KafkaConsumerThread(self.kafka_address, self.kafka_port, topic=["object_detection"], sync_queue=self.sync_queue)

        if estimator is None:
            self.log.error("ERROR! Estimator is not set!")

        # Operating Mode
        self.op_mode = "detection"
        self.target = None

    def start(self):
        if callable(self.estimator):
            self.estimator = self.estimator()

        self.log.info(f"Loading model")
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"{self.model_path} miss")
        else:
            self.estimator.load(self.model_path)

        self.log.info("Evaluating model")
        self.estimator.evaluate()

        StatusSyncThread(self.update_operational_mode)

        if self.kafka_enabled:
            self.log.debug("Creating Apache Kafka thread to fetch data")
            self.fetch_data()
        else:
            self.log.debug("Starting default REST webservice/s")

            self.cloud = ReID(service_name=self.job_name,host=self.remote_ip, port=self.remote_port)
            app_server = FEServer(model=self, servername=self.job_name,host=self.local_ip, http_port=self.local_port)

            self.queue = queue.Queue()
            threading.Thread(target=self.fetch_data, daemon=True).start()

            app_server.start()

    def fetch_data(self):
        while True:
            token = self.sync_queue.get()
            self.log.debug(f'Data consumed')
            try:
                self.inference(token)
            except Exception as e:
                msg = f"Error processing token {token}: {e}" 
                self.log.error((msg[:60] + '..' + msg[len(msg)-40:-1]) if len(msg) > 60 else msg)

            self.sync_queue.task_done()

    def put_data(self, data):
        self.sync_queue.put(data)
        self.log.debug("Data deposited")

    def inference(self, data=None, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        res = self.estimator.predict(data, **kwargs)
        fe_result = deepcopy(res)

        if callback_func:
            res = callback_func(res)

        if fe_result != []:
            with FTimer(f"upload_fe_results"):
                if self.kafka_enabled:
                    cres = self.producer.write_result(fe_result)
                else:
                    cres = self.cloud.reid([fe_result], post_process=post_process, **kwargs)

        return [None, cres, fe_result, None]


    def update_operational_mode(self, status):
        self.log.debug("Configuration update triggered")

        if status == None:
            return
        
        if self.op_mode != status['op_mode']:
            self.log.info(f"{status['op_mode']} mode activated!")
            self.op_mode = status['op_mode']

            if self.op_mode == "detection":
                self.target = None
                return

            if self.target != status['target'] and self.op_mode != "detection":
                self.target = status['target']
                json_data = json.dumps(self.target)
                img_bytes = base64.b64decode(json_data.encode('utf-8'))

                # convert bytes data to PIL Image object
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

                # PIL image object to numpy array
                img_arr = np.asarray(img)      

                data = [ img_arr, 1.0, 0, 0, 1 ]
                self.inference(data, post_process=None, new_target=True)
            else:
                self.log.debug("Target unchanged")


class ObjectDetector(JobBase):
    """
   ObjectDetector service.
   """

    def __init__(self, estimator=None, config=None):
        super(ObjectDetector, self).__init__(
            estimator=estimator, config=config)
        self.log.info("Loading ObjectDetector module")
        
        self.local_ip = get_host_ip()

        self.remote_ip = self.get_parameters("FE_MODEL_BIND_URL", self.local_ip)
        self.port = int(self.get_parameters("FE_MODEL_BIND_PORT", "6000"))

        self.kafka_enabled = bool(distutils.util.strtobool(self.get_parameters("KAFKA_ENABLED", "False")))

        if estimator is None:
            self.log.error("ERROR! Estimator is not set!")

        if self.kafka_enabled:
            self.log.debug("Kafka support enabled in YAML file")
            self.kafka_address = self.get_parameters("KAFKA_BIND_IPS", ["7.182.9.110"])
            self.kafka_port = self.get_parameters("KAFKA_BIND_PORTS", [32669])

            if isinstance(self.kafka_address, str):
                self.log.debug(f"Parsing string received from K8s controller {self.kafka_address},{self.kafka_port}")
                self.kafka_address = self.kafka_address.split("|")
                self.kafka_port = self.kafka_port.split("|")

            self.producer = KafkaProducer(self.kafka_address, self.kafka_port, topic=["object_detection"])
        
        # Operating Mode
        self.op_mode = "detection"
        self.target = None

        self.start()

    def start(self):
        if callable(self.estimator):
            self.estimator = self.estimator()

        self.log.info("Loading model")
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"Cannot find model: {self.model_path}")
        else:
            self.estimator.load(self.model_path)

        self.log.info("Evaluating model")
        self.estimator.evaluate()

        StatusSyncThread(self.update_operational_mode)

        if not self.kafka_enabled:
            self.log.debug("Starting default REST webservice/s")
            self.edge = FE(service_name=self.job_name,host=self.remote_ip, port=self.port)

    def inference(self, data=None, post_process=None, **kwargs):
        callback_func = None
        cres = None
        
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        detection_result = self.estimator.predict(data, **kwargs)

        if callback_func:
            detection_result = callback_func(detection_result)

        if detection_result != None and len(detection_result) > 0:
            with FTimer(f"upload_bboxes"):
                if self.kafka_enabled:
                    cres = self.producer.write_result(detection_result)
                else:
                    cres = self.edge.feature_extraction([detection_result], post_process=post_process, **kwargs)

        return [cres, detection_result]

    def update_operational_mode(self, status):
        self.log.debug("Configuration update triggered")

        if status == None:
            return

        self.op_mode = status['op_mode']

class StatusSyncThread(threading.Thread):

    def __init__(self, callback, controller_address="http://7.182.9.110:27345/sedna/get_app_details", update_interval=10):
        super().__init__()
        LOGGER.info("Creating StatusSyncThread")

        self.callback = callback
        self.controller_address = controller_address
        self.update_interval = update_interval
        self.daemon = True

        self.start()

    def sync_configuration(self):
        LOGGER.debug(f'Attempting to synchronize the pod configuration')
        try:
            status = requests.get(self.controller_address)
            LOGGER.debug(f'Current status retrieved')
            return json.loads(status.content)

        except Exception as ex:
            LOGGER.error(f'Unable to retrieve sync status, using fallback value. [{ex}]')

        return None


    def run(self):
        LOGGER.info("Start status synchronization thread")
        while 1:
            with FTimer("status_update"):
                status = self.sync_configuration()
            self.callback(status) #pass a callback function to the thread, it's better
            time.sleep(self.update_interval)

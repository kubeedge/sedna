import abc
import json
import logging
import os
import threading
import time

import cv2
import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from flask import Flask, request

import sedna
from sedna.common.config import BaseConfig
from sedna.common.constant import K8sResourceKind
from sedna.hard_example_mining import CrossEntropyFilter, IBTFilter, \
    ThresholdFilter
from sedna.joint_inference.data import ServiceInfo
from sedna.lc_client import LCClient

LOG = logging.getLogger(__name__)


class BigModelConfig(BaseConfig):
    def __init__(self):
        BaseConfig.__init__(self)
        self.bind_ip = os.getenv("BIG_MODEL_BIND_IP", "0.0.0.0")
        self.bind_port = (
            int(os.getenv("BIG_MODEL_BIND_PORT", "5000"))
        )


class LittleModelConfig(BaseConfig):
    def __init__(self):
        BaseConfig.__init__(self)


class BigModelClientConfig:
    def __init__(self):
        self.ip = os.getenv("BIG_MODEL_IP")
        self.port = int(os.getenv("BIG_MODEL_PORT", "5000"))


class BaseModel:
    """Model abstract class.

    :param preprocess: function before inference
    :param postprocess: function after inference
    :param input_shape: input shape
    :param create_input_feed: the function of creating input feed
    :param create_output_fetch: the function fo creating output fetch
    """

    def __init__(self, preprocess=None, postprocess=None, input_shape=(0, 0),
                 create_input_feed=None, create_output_fetch=None):
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.input_shape = input_shape
        if create_input_feed is None or create_output_fetch is None:
            raise RuntimeError("Please offer create_input_feed "
                               "and create_output_fetch function")
        self.create_input_feed = create_input_feed
        self.create_output_fetch = create_output_fetch

    @abc.abstractmethod
    def _load_model(self):
        pass

    @abc.abstractmethod
    def inference(self, img_data):
        pass


class BigModelClient:
    """Remote big model service, which interacts with the cloud big model."""
    _retry = 5
    _retry_interval_seconds = 1

    def __init__(self):
        self.config = BigModelClientConfig()
        self.big_model_endpoint = "http://{0}:{1}".format(
            self.config.ip,
            self.config.port
        )

    def _load_model(self):
        pass

    def inference(self, img_data):
        """Use the remote big model server to inference."""
        _, encoded_image = cv2.imencode(".jpeg", img_data)
        files = {"images": encoded_image}
        error = None
        for i in range(BigModelClient._retry):
            try:
                res = requests.post(self.big_model_endpoint, timeout=5,
                                    files=files)
                if res.status_code < 300:
                    return res.json().get("data")
                else:
                    LOG.error(f"send request to {self.big_model_endpoint} "
                              f"failed, status is {res.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                error = e
                time.sleep(BigModelClient._retry_interval_seconds)

        LOG.error(f"send request to {self.big_model_endpoint} failed, "
                  f"error is {error}, retry times: {BigModelClient._retry}")
        return None


class TSBigModelService(BaseModel):
    """Large model services implemented based on TensorFlow.
    Provides RESTful interfaces for large-model inference.
    """

    def __init__(self, preprocess=None, postprocess=None, input_shape=(0, 0),
                 create_input_feed=None, create_output_fetch=None):
        BaseModel.__init__(self, preprocess, postprocess, input_shape,
                           create_input_feed, create_output_fetch)
        self.config = BigModelConfig()

        self.input_shape = input_shape
        self._load_model()

        self.app = Flask(__name__)
        self.register()
        self.app.run(host=self.config.bind_ip,
                     port=self.config.bind_port)

    def register(self):
        @self.app.route('/', methods=['POST'])
        def inference():
            f = request.files.get('images')
            image = Image.open(f)
            image = image.convert("RGB")
            img_data, org_img_shape = self.preprocess(image, self.input_shape)
            data = self.inference(img_data)
            result = self.postprocess(data, org_img_shape)
            # encapsulate the user result
            data = {"data": result}
            return json.dumps(data)

    def _load_model(self):
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.InteractiveSession(graph=self.graph)

        with tf.io.gfile.GFile(self.config.model_url, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')
        LOG.info(f"Import yolo model from {self.config.model_url} end .......")

    def inference(self, img_data):
        input_feed = self.create_input_feed(self.sess, img_data)
        output_fetch = self.create_output_fetch(self.sess)

        return self.sess.run(output_fetch, input_feed)


class TSLittleModel(BaseModel):
    """Little model services implemented based on TensorFlow.
    Provides RESTful interfaces for large-model inference.
    """

    def __init__(self, preprocess=None, postprocess=None, input_shape=(0, 0),
                 create_input_feed=None, create_output_fetch=None):
        BaseModel.__init__(self, preprocess, postprocess, input_shape,
                           create_input_feed, create_output_fetch)

        self.config = LittleModelConfig()

        graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.session = tf.Session(graph=graph, config=config)
        self._load_model()

    def _load_model(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                with tf.gfile.FastGFile(self.config.model_url, 'rb') as handle:
                    LOG.info(f"Load model {self.config.model_url}, "
                             f"ParseFromString start .......")
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(handle.read())
                    LOG.info("ParseFromString end .......")

                    tf.import_graph_def(graph_def, name='')
                    LOG.info("Import_graph_def end .......")

        LOG.info("Import model from pb end .......")

    def inference(self, img_data):
        img_data_np = np.array(img_data)
        with self.session.as_default():
            new_image = self.preprocess(img_data_np, self.input_shape)
            input_feed = self.create_input_feed(self.session, new_image,
                                                img_data_np)
            output_fetch = self.create_output_fetch(self.session)
            output = self.session.run(output_fetch, input_feed)
            if self.postprocess:
                output = self.postprocess(output)
            return output


class LCReporter(threading.Thread):
    """Inherited thread, which is an entity that periodically report to
    the lc.
    """

    def __init__(self):
        threading.Thread.__init__(self)

        # the value of statistics
        self.inference_number = 0
        self.hard_example_number = 0
        self.period_interval = int(os.getenv("LC_PERIOD", "30"))
        # The system resets the period_increment after sending the messages to
        # the LC. If the period_increment is 0 in the current period,
        # the system does not send the messages to the LC.
        self.period_increment = 0
        self.lock = threading.Lock()

    def update_for_edge_inference(self):
        self.lock.acquire()
        self.inference_number += 1
        self.period_increment += 1
        self.lock.release()

    def update_for_collaboration_inference(self):
        self.lock.acquire()
        self.inference_number += 1
        self.hard_example_number += 1
        self.period_increment += 1
        self.lock.release()

    def run(self):
        while True:

            info = ServiceInfo()
            info.startTime = time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime())

            time.sleep(self.period_interval)
            if self.period_increment == 0:
                LOG.debug("period increment is zero, skip report")
                continue
            info.updateTime = time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime())
            info.inferenceNumber = self.inference_number
            info.hardExampleNumber = self.hard_example_number
            info.uploadCloudRatio = (
                self.hard_example_number / self.inference_number
            )
            message = {
                "name": BaseConfig.worker_name,
                "namespace": BaseConfig.namespace,
                "ownerName": BaseConfig.service_name,
                "ownerKind": K8sResourceKind.JOINT_INFERENCE_SERVICE.value,
                "kind": "inference",
                "ownerInfo": info.__dict__,
                "results": []
            }
            LCClient.send(BaseConfig.worker_name, message)
            self.period_increment = 0


class InferenceResult:
    """The Result class for joint inference

    :param is_hard_example: `True` means a hard example, `False` means not a
        hard example
    :param final_result: the final inference result
    :param hard_example_edge_result: the edge little model inference result of
        hard example
    :param hard_example_cloud_result: the cloud big model inference result of
        hard example
    """

    def __init__(self, is_hard_example, final_result,
                 hard_example_edge_result, hard_example_cloud_result):
        self.is_hard_example = is_hard_example
        self.final_result = final_result
        self.hard_example_edge_result = hard_example_edge_result
        self.hard_example_cloud_result = hard_example_cloud_result


class JointInference:
    """Class provided for external systems for model joint inference.

    :param little_model: the little model entity for edge inference
    :param hard_example_mining_algorithm: the algorithm for judging hard
        example
    """

    def __init__(self, little_model: BaseModel,
                 hard_example_mining_algorithm=None):
        self.little_model = little_model
        self.big_model = BigModelClient()
        # TODO how to deal process use-defined cloud_offload_algorithm,
        # especially parameters
        if hard_example_mining_algorithm is None:
            hem_name = BaseConfig.hem_name

            if hem_name == "IBT":
                threshold_box = float(sedna.context.get_hem_parameters(
                    "threshold_box", 0.5
                ))
                threshold_img = float(sedna.context.get_hem_parameters(
                    "threshold_img", 0.5
                ))
                hard_example_mining_algorithm = IBTFilter(threshold_img,
                                                          threshold_box)
            elif hem_name == "CrossEntropy":
                threshold_cross_entropy = float(
                    sedna.context.get_hem_parameters(
                        "threshold_cross_entropy", 0.5
                    )
                )
                hard_example_mining_algorithm = CrossEntropyFilter(
                    threshold_cross_entropy)
            else:
                hard_example_mining_algorithm = ThresholdFilter()

        self.hard_example_mining_algorithm = hard_example_mining_algorithm

        self.lc_reporter = LCReporter()
        self.lc_reporter.setDaemon(True)
        self.lc_reporter.start()

    def inference(self, img_data) -> InferenceResult:
        """Image inference function."""
        img_data_pre = img_data
        edge_result = self.little_model.inference(img_data_pre)
        is_hard_example = self.hard_example_mining_algorithm.hard_judge(
            edge_result
        )
        if not is_hard_example:
            LOG.debug("not hard example, use edge result directly")
            self.lc_reporter.update_for_edge_inference()
            return InferenceResult(False, edge_result, None, None)

        cloud_result = self._cloud_inference(img_data)
        if cloud_result is None:
            LOG.warning("retrieve cloud infer service failed, use edge result")
            self.lc_reporter.update_for_edge_inference()
            return InferenceResult(True, edge_result, edge_result, None)
        else:
            LOG.debug(f"retrieve cloud infer service success, use cloud "
                      f"result, cloud result:{cloud_result}")
            self.lc_reporter.update_for_collaboration_inference()
            return InferenceResult(True, cloud_result, edge_result,
                                   cloud_result)

    def _cloud_inference(self, img_rgb):
        return self.big_model.inference(img_rgb)


def _get_or_default(parameter, default):
    value = sedna.context.get_parameters(parameter)
    return value if value else default

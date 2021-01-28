import json
import logging
import os

from sedna.common.config import BaseConfig
from sedna.common.constant import K8sResourceKind
from sedna.common.utils import remove_path_prefix, model_layer_flatten, \
    model_layer_reshape
from sedna.federated_learning.aggregator import AggregationClient
from sedna.federated_learning.data import AggregationData
from sedna.lc_client import LCClient

LOG = logging.getLogger(__name__)


class FederatedConfig(BaseConfig):
    def __init__(self):
        BaseConfig.__init__(self)
        self.agg_ip = os.getenv("AGG_IP", "0.0.0.0")
        self.agg_port = int(os.getenv("AGG_PORT", "7363"))


def train(model,
          x,
          y,
          epochs,
          batch_size,
          loss,
          optimizer,
          metrics,
          validation_split=0.2,
          aggregation_algorithm='fed_avg'):
    """
    :param model: A predefined neural network based on keras. Currently only
        keras model is supported.
    :type model: class `keras.models.Sequential`
    :param x: Feature data.
    :type x: class `numpy.ndarray`
    :param y: Label data.
    :type y: class `numpy.ndarray`
    :param epochs: The number of epochs for training the model.
    :type epochs: int
    :param batch_size: number of samples in a training.
    :type batch_size: int
    :param loss: loss function. Currently only losses function defined in
        keras are supported.
    :type loss: class `tf.keras.losses.Loss` instance
    :param optimizer: Currently only optimizer defined in keras are supported.
    :type optimizer: Optimizers defined in `tf.keras.optimizers`
    :param metrics: List of metrics to be evaluated by the model during
        training and testing.
    :type metrics: class `tf.keras.metrics.Metric` instance
    :param validation_split: Fraction of the training data to be used as
        validation data.
    :type validation_split: float
    :param aggregation_algorithm: Federated learning algorithm for edge.
        Default is FedAvg, more algorithms are being integrated.
    :type aggregation_algorithm: str
    :return: The compiled `tf.keras.Model`.
    """

    fl_config = FederatedConfig()
    transmitter = AggregationClient(ip=fl_config.agg_ip,
                                    port=fl_config.agg_port)

    LOG.info("transmitter started!")

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    round_number = 0

    while True:
        round_number += 1
        LOG.info(f"start training, round_number={round_number}")
        history = model.fit(x, y,
                            epochs=int(epochs),
                            batch_size=int(batch_size),
                            verbose=0,
                            validation_split=validation_split)

        data = AggregationData()
        data.task_id = fl_config.job_name
        data.worker_id = fl_config.worker_name
        data.sample_count = len(x)
        data.shapes = [tuple(layer.shape) for layer in model.get_weights()]
        data.flatten_weights = model_layer_flatten(model.get_weights())

        received: AggregationData = transmitter.update_weights(data)
        model.set_weights(model_layer_reshape(received.flatten_weights,
                                              received.shapes))

        # determine whether to exit training
        if not received.exit_flag:
            _report_task_info(received.task_info, 'training', history)
        else:
            _report_task_info(received.task_info, 'completed', history)
            LOG.info(f"exit training for task [{fl_config.job_name}]")
            return model


def _report_task_info(task_info, status, history):
    fl_config = FederatedConfig()
    if task_info is None:
        LOG.info(f"task info is None, no need to report to lc.")
        return
    ckpt_model_url = remove_path_prefix(fl_config.model_url,
                                        fl_config.data_path_prefix)
    pb_model_url = remove_path_prefix(
        os.path.join(fl_config.model_url, 'model.pb'),
        fl_config.data_path_prefix)
    ckpt_result = {
        "format": "ckpt",
        "url": ckpt_model_url,
        "metrics": history.history
    }

    pb_result = {
        "format": "pb",
        "url": pb_model_url,
        "metrics": history.history
    }

    results = [ckpt_result, pb_result]
    message = {
        "name": fl_config.worker_name,
        "namespace": fl_config.namespace,
        "ownerName": fl_config.job_name,
        "ownerKind": K8sResourceKind.FEDERATED_LEARNING_JOB.value,
        "kind": "train",
        "status": status,
        "ownerInfo": json.loads(task_info),
        "results": results
    }
    LCClient.send(fl_config.worker_name, message)

import logging

import os
import tensorflow as tf

import sedna
from sedna.common.config import BaseConfig
from sedna.common.constant import K8sResourceKindStatus, K8sResourceKind
from sedna.common.utils import clean_folder, remove_path_prefix
from sedna.hard_example_mining import CrossEntropyFilter, IBTFilter, \
    ThresholdFilter
from sedna.joint_inference import TSLittleModel
from sedna.lc_client import LCClient

LOG = logging.getLogger(__name__)


class IncrementalConfig(BaseConfig):
    def __init__(self):
        BaseConfig.__init__(self)
        self.model_urls = os.getenv("MODEL_URLS")
        self.base_model_url = os.getenv("BASE_MODEL_URL")
        self.saved_model_name = "model.pb"


def train(model, train_data, epochs, batch_size, class_names, input_shape,
          obj_threshold, nms_threshold):
    """The train endpoint of incremental learning.

    :param model: the train model
    :param train_data: the data use for train
    :param epochs: the number of epochs for training the model
    :param batch_size: the number of samples in a training
    :param class_names:
    :param input_shape:
    :param obj_threshold:
    :param nms_threshold:
    """
    il_config = IncrementalConfig()

    clean_folder(il_config.model_url)
    model.train(train_data, [])  # validation data is empty.
    tf.reset_default_graph()
    model.save_model_pb(il_config.saved_model_name)

    ckpt_model_url = remove_path_prefix(il_config.model_url,
                                        il_config.data_path_prefix)
    pb_model_url = remove_path_prefix(
        os.path.join(il_config.model_url, il_config.saved_model_name),
        il_config.data_path_prefix)

    # TODO delete metrics whether affect lc
    ckpt_result = {
        "format": "ckpt",
        "url": ckpt_model_url,
    }

    pb_result = {
        "format": "pb",
        "url": pb_model_url,
    }

    results = [ckpt_result, pb_result]

    message = {
        "name": il_config.worker_name,
        "namespace": il_config.namespace,
        "ownerName": il_config.job_name,
        "ownerKind": K8sResourceKind.INCREMENTAL_JOB.value,
        "kind": "train",
        "status": K8sResourceKindStatus.COMPLETED.value,
        "results": results
    }
    LCClient.send(il_config.worker_name, message)


def evaluate(model, test_data, class_names, input_shape):
    """The evaluation endpoint of incremental job.

    :param model: the model used for evaluation
    :param test_data:
    :param class_names:
    :param input_shape: the input shape of model
    """
    il_config = IncrementalConfig()

    results = []
    for model_url in il_config.model_urls.split(';'):
        precision, recall, all_precisions, all_recalls = model(
            model_path=model_url,
            test_dataset=test_data,
            class_names=class_names,
            input_shape=input_shape)

        result = {
            "format": "pb",
            "url": remove_path_prefix(model_url, il_config.data_path_prefix),
            "metrics": {
                "recall": recall,
                "precision": precision
            }
        }
        results.append(result)

    message = {
        "name": il_config.worker_name,
        "namespace": il_config.namespace,
        "ownerName": il_config.job_name,
        "ownerKind": K8sResourceKind.INCREMENTAL_JOB.value,
        "kind": "eval",
        "status": K8sResourceKindStatus.COMPLETED.value,
        "results": results
    }

    LCClient.send(il_config.worker_name, message)


class TSModel(TSLittleModel):
    def __init__(self, preprocess=None, postprocess=None, input_shape=(0, 0),
                 create_input_feed=None, create_output_fetch=None):
        TSLittleModel.__init__(self, preprocess, postprocess, input_shape,
                               create_input_feed, create_output_fetch)


class InferenceResult:
    def __init__(self, is_hard_example, infer_result):
        self.is_hard_example = is_hard_example
        self.infer_result = infer_result


class Inference:
    def __init__(self, model: TSModel, hard_example_mining_algorithm=None):
        if hard_example_mining_algorithm is None:
            hem_name = BaseConfig.hem_name

            if hem_name == "IBT":
                threshold_box = float(sedna.context.get_hem_parameters(
                    "threshold_box", 0.8
                ))
                threshold_img = float(sedna.context.get_hem_parameters(
                    "threshold_img", 0.8
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
        self.model = model

    def inference(self, img_data) -> InferenceResult:
        result = self.model.inference(img_data)
        rsl = deal_infer_rsl(result)
        is_hard_example = self.hard_example_mining_algorithm.hard_judge(rsl)
        if is_hard_example:
            return InferenceResult(True, result)
        else:
            return InferenceResult(False, result)


def deal_infer_rsl(model_output):
    all_classes, all_scores, all_bboxes = model_output
    rsl = []
    for c, s, bbox in zip(all_classes, all_scores, all_bboxes):
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1], bbox[0], bbox[3], bbox[2]
        rsl.append(bbox.tolist() + [s, c])

    return rsl

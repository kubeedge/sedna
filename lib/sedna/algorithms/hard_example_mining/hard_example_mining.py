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

"""Hard Example Mining Algorithms"""

import abc
import math
import os
import random
from sedna.common.class_factory import ClassFactory, ClassType
import logging

# Only import torch and transformers when BACKEND_TYPE is TORCH
BACKEND_TYPE = os.environ.get('BACKEND_TYPE', '').upper()
if BACKEND_TYPE == 'TORCH':
    import torch
    from transformers import pipeline

__all__ = ['ThresholdFilter', 'CrossEntropyFilter', 'IBTFilter',
           'RandomFilter', 'CloudOnlyFilter', 'EdgeOnlyFilter']

# Only add BertRouterFilter to __all__ when BACKEND_TYPE is TORCH
if BACKEND_TYPE == 'TORCH':
    __all__.append('BertRouterFilter')

LOG = logging.getLogger(__name__)


class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define unified interface."""

    def __call__(self, infer_result=None):
        """
        predict function, judge the sample is hard or not.

        Parameters
        ----------
        infer_result : array_like
            prediction result

        Returns
        -------
        is_hard_sample : bool
            `True` means hard sample, `False` means not.
        """
        raise NotImplementedError

    @classmethod
    def data_check(cls, data):
        """Check the data in [0,1]."""
        return 0 <= float(data) <= 1


@ClassFactory.register(ClassType.HEM, alias="Threshold")
class ThresholdFilter(BaseFilter, abc.ABC):
    """
    **Object detection** Hard samples discovery methods named `Threshold`

    Parameters
    ----------
    threshold: float
        hard coefficient threshold score to filter img, default to 0.5.
    """

    def __init__(self, threshold: float = 0.5, **kwargs):
        self.threshold = float(threshold)

    def __call__(self, infer_result=None) -> bool:
        # if invalid input, return False
        if not (infer_result
                and all(map(lambda x: len(x) > 4, infer_result))):
            return False

        image_score = 0

        for bbox in infer_result:
            image_score += bbox[4]

        average_score = image_score / (len(infer_result) or 1)
        return average_score < self.threshold


@ClassFactory.register(ClassType.HEM, alias="CrossEntropy")
class CrossEntropyFilter(BaseFilter, abc.ABC):
    """
    **Object detection** Hard samples discovery methods named `CrossEntropy`

    Parameters
    ----------
    threshold_cross_entropy: float
        hard coefficient threshold score to filter img, default to 0.5.
    """

    def __init__(self, threshold_cross_entropy=0.5, **kwargs):
        self.threshold_cross_entropy = float(threshold_cross_entropy)

    def __call__(self, infer_result=None) -> bool:
        """judge the img is hard sample or not.

        Parameters
        ----------
        infer_result: array_like
            prediction classes list, such as
            [class1-score, class2-score, class2-score,....],
            where class-score is the score corresponding to the class,
            class-score value is in [0,1], who will be ignored if its
            value not in [0,1].

        Returns
        -------
        is hard sample: bool
            `True` means hard sample, `False` means not.
        """

        if not infer_result:
            # if invalid input, return False
            return False

        log_sum = 0.0
        data_check_list = [class_probability for class_probability
                           in infer_result
                           if self.data_check(class_probability)]

        if len(data_check_list) != len(infer_result):
            return False

        for class_data in data_check_list:
            log_sum += class_data * math.log(class_data)
        confidence_score = 1 + 1.0 * log_sum / math.log(
            len(infer_result))
        return confidence_score < self.threshold_cross_entropy


@ClassFactory.register(ClassType.HEM, alias="IBT")
class IBTFilter(BaseFilter, abc.ABC):
    """
    **Object detection** Hard samples discovery methods named `IBT`

    Parameters
    ----------
    threshold_img: float
        hard coefficient threshold score to filter img, default to 0.5.
    threshold_box: float
        threshold_box to calculate hard coefficient, formula  is hard
        coefficient = number(prediction_boxes less than threshold_box) /
        number(prediction_boxes)
    """

    def __init__(self, threshold_img=0.5, threshold_box=0.5, **kwargs):
        self.threshold_box = float(threshold_box)
        self.threshold_img = float(threshold_img)

    def __call__(self, infer_result=None) -> bool:
        """Judge the img is hard sample or not.

        Parameters
        ----------
        infer_result: array_like
            prediction boxes list, such as [bbox1, bbox2, bbox3,....],
            where bbox = [xmin, ymin, xmax, ymax, score, label]
            score should be in [0,1], who will be ignored if its value not
            in [0,1].

        Returns
        -------
        is hard sample: bool
            `True` means hard sample, `False` means not.
        """

        if not (infer_result
                and all(map(lambda x: len(x) > 4, infer_result))):
            # if invalid input, return False
            return False

        data_check_list = [bbox[4] for bbox in infer_result
                           if self.data_check(bbox[4])]
        if len(data_check_list) != len(infer_result):
            return False

        confidence_score_list = [
            float(box_score) for box_score in data_check_list
            if float(box_score) <= self.threshold_box]
        return (len(confidence_score_list) / len(infer_result)
                >= (1 - self.threshold_img))


@ClassFactory.register(ClassType.HEM, alias="Random")
class RandomFilter(BaseFilter):
    """judge a image is hard example or not randomly

            Parameters
            ----------
            random_ratio: int
                value: between 0 and 1
                with a model having very high accuracy like 98%, use this
                function to define an input is hard example or not. just
                a meaningless but needed function in sedna incremental learning
                inference

            Returns
            -------
            is hard sample: bool
                `True` means hard sample, `False` means not.
            """

    def __init__(self, random_ratio=0.3, **kwargs):
        self.random_ratio = random_ratio

    def __call__(self, *args, **kwargs):
        if random.uniform(0, 1) < self.random_ratio:
            return True
        return False


@ClassFactory.register(ClassType.HEM, alias="CloudOnly")
class CloudOnlyFilter(BaseFilter):
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return True


@ClassFactory.register(ClassType.HEM, alias="EdgeOnly")
class EdgeOnlyFilter(BaseFilter):
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return False


# Only define BertRouterFilter when BACKEND_TYPE is TORCH
if BACKEND_TYPE == 'TORCH':
    @ClassFactory.register(ClassType.HEM, alias="BertRouter")
    class BertRouterFilter(BaseFilter, abc.ABC):
        def __init__(self, **kwargs):
            """Initialize the BERTFilter.

            Parameters
            ----------
            kwargs: dict
                Possible kwargs are:
                - `model`: str, default "routellm/bert". The model to use.
                - `task`: str, default "text-classification". Task to use.
                - `max_length`: int, default 512. Max length of the input.
            """

            self.model = kwargs.get("model", "routellm/bert")
            self.task = kwargs.get("task", "text-classification")
            self.max_length = int(kwargs.get("max_length", 512))
            self.device = kwargs.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu")
            self.threshold = float(kwargs.get("threshold", 0.5))

            try:
                self.classifier = pipeline(
                    self.task, model=self.model, device=self.device)
            except Exception as e:
                LOG.error(f"Failed to initialize the pipeline: {e}")
                raise RuntimeError(
                    "Pipeline initialization failed. "
                    "Please check the model and task parameters.")

        def _text_classification_postprocess(self, result):
            """Postprocess the text classification result

            Parameters
            ----------
            result : list
                The result from the classifier. Example:
                ```
                [{"label": "LABEL_0", "score": 0.5},
                {"label": "LABEL_1", "score": 0.4},
                {"label": "LABEL_2", "score": 0.1}]

            Returns
            -------
            bool
                `True` means hard sample, `False` means not.
            """

            res = {item["label"]: item["score"] for item in result}
            scaled_score = res["LABEL_0"] / (res["LABEL_0"] + res["LABEL_1"])

            label = "LABEL_0" if scaled_score >= self.threshold else "LABEL_1"
            return False if label == "LABEL_0" else True

        def _predict(self, data):
            """Predict the data label

            Parameters
            ----------
            data : dict
                See format at BaseLLM's `inference()`.

            Returns
            -------
            bool
                `True` means hard sample, `False` means not.

            Raises
            ------
            NotImplementedError
                If the task is not supported
            """

            if self.task == "text-classification":
                result = self.classifier(data, top_k=None)
                is_hard_sample = \
                    self._text_classification_postprocess(result[0])
            else:
                raise NotImplementedError

            return is_hard_sample

        def _preprocess(self, data):
            """Preprocess the data

            Parameters
            ----------
            data : dict
                See format at BaseLLM's `inference()`.

            Returns
            -------
            str
                query string
            """
            query = data.get("query")
            if "query" in query:
                return query["query"][:self.max_length]
            else:
                return query[:self.max_length]

        def cleanup(self):
            """Release the classifier model
            """
            del self.classifier

        def __call__(self, data=None) -> bool:
            # data = self._preprocess(data)
            return self._predict(data)

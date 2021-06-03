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
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('ThresholdFilter', 'CrossEntropyFilter', 'IBTFilter')


class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define unified interface."""

    def __call__(self, infer_result=None):
        """predict function, and it must be implemented by
        different methods class.

        :param infer_result: prediction result
        :return: `True` means hard sample, `False` means not a hard sample.
        """
        raise NotImplementedError

    @classmethod
    def data_check(cls, data):
        """Check the data in [0,1]."""
        return 0 <= float(data) <= 1


@ClassFactory.register(ClassType.HEM, alias="Threshold")
class ThresholdFilter(BaseFilter, abc.ABC):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = float(threshold)

    def __call__(self, infer_result=None):
        """
        :param infer_result: [N, 6], (x0, y0, x1, y1, score, class)
        :return: `True` means hard sample, `False` means not a hard sample.
        """
        if not infer_result:
            return True

        image_score = 0
        for bbox in infer_result:
            image_score += bbox[4]

        average_score = image_score / (len(infer_result) or 1)
        return average_score < self.threshold


@ClassFactory.register(ClassType.HEM, alias="CrossEntropy")
class CrossEntropyFilter(BaseFilter, abc.ABC):
    """ Implement the hard samples discovery methods named IBT
    (image-box-thresholds).

    :param threshold_cross_entropy: threshold_cross_entropy to filter img,
                        whose hard coefficient is less than
                        threshold_cross_entropy. And its default value is
                        threshold_cross_entropy=0.5
    """

    def __init__(self, threshold_cross_entropy=0.5, **kwargs):
        self.threshold_cross_entropy = float(threshold_cross_entropy)

    def __call__(self, infer_result=None):
        """judge the img is hard sample or not.

        :param infer_result:
            prediction classes list,
                such as [class1-score, class2-score, class2-score,....],
                where class-score is the score corresponding to the class,
                class-score value is in [0,1], who will be ignored if its value
                 not in [0,1].
        :return: `True` means a hard sample, `False` means not a hard sample.
        """
        if infer_result is None:
            return False
        elif len(infer_result) == 0:
            return False
        else:
            log_sum = 0.0
            data_check_list = [class_probability for class_probability
                               in infer_result
                               if self.data_check(class_probability)]
            if len(data_check_list) == len(infer_result):
                for class_data in data_check_list:
                    log_sum += class_data * math.log(class_data)
                confidence_score = 1 + 1.0 * log_sum / math.log(
                    len(infer_result))
                return confidence_score < self.threshold_cross_entropy
            else:
                return False


@ClassFactory.register(ClassType.HEM, alias="IBT")
class IBTFilter(BaseFilter, abc.ABC):
    """Implement the hard samples discovery methods named IBT
        (image-box-thresholds).

    :param threshold_img: threshold_img to filter img, whose hard coefficient
        is less than threshold_img.
    :param threshold_box: threshold_box to calculate hard coefficient, formula
        is hard coefficient = number(prediction_boxes less than
            threshold_box)/number(prediction_boxes)
    """

    def __init__(self, threshold_img=0.5, threshold_box=0.5, **kwargs):
        self.threshold_box = float(threshold_box)
        self.threshold_img = float(threshold_img)

    def __call__(self, infer_result=None):
        """Judge the img is hard sample or not.

        :param infer_result:
            prediction boxes list,
                such as [bbox1, bbox2, bbox3,....],
                where bbox = [xmin, ymin, xmax, ymax, score, label]
                score should be in [0,1], who will be ignored if its value not
                in [0,1].
        :return: `True` means a hard sample, `False` means not a hard sample.
        """
        if infer_result is None:
            return False
        elif len(infer_result) == 0:
            return False
        else:
            data_check_list = [bbox[4] for bbox in infer_result
                               if self.data_check(bbox[4])]
            if len(data_check_list) == len(infer_result):
                confidence_score_list = [
                    float(box_score) for box_score in data_check_list
                    if float(box_score) <= self.threshold_box]
                if (len(confidence_score_list) / len(infer_result)) \
                        >= (1 - self.threshold_img):
                    return True
                else:
                    return False
            else:
                return False

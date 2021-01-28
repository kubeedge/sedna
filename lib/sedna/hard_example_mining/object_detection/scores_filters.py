import logging

from sedna.hard_example_mining import BaseFilter
from sedna.hard_example_mining.hard_example_helpers import data_check

logger = logging.getLogger(__name__)


class IBTFilter(BaseFilter):
    """Implement the hard samples discovery methods named IBT
        (image-box-thresholds).

    :param threshold_img: threshold_img to filter img, whose hard coefficient
        is less than threshold_img.
    :param threshold_box: threshold_box to calculate hard coefficient, formula
        is hard coefficient = number(prediction_boxes less than
            threshold_box)/number(prediction_boxes)
    """

    def __init__(self, threshold_img=0.5, threshold_box=0.5):
        self.threshold_box = threshold_box
        self.threshold_img = threshold_img

    def hard_judge(self, infer_result=None):
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
            logger.warning(f'infer result is invalid, value: {infer_result}!')
            return False
        elif len(infer_result) == 0:
            return False
        else:
            data_check_list = [bbox[4] for bbox in infer_result
                               if data_check(bbox[4])]
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
                logger.warning(
                    "every value of infer_result should be in [0,1], "
                    f"your data is {infer_result}")
                return False

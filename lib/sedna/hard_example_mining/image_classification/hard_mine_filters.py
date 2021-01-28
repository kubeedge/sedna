import logging
import math

from sedna.hard_example_mining import BaseFilter
from sedna.hard_example_mining.hard_example_helpers import data_check

logger = logging.getLogger(__name__)


class CrossEntropyFilter(BaseFilter):
    """ Implement the hard samples discovery methods named IBT
    (image-box-thresholds).

    :param threshold_cross_entropy: threshold_cross_entropy to filter img,
                        whose hard coefficient is less than
                        threshold_cross_entropy. And its default value is
                        threshold_cross_entropy=0.5
    """

    def __init__(self, threshold_cross_entropy=0.5):
        self.threshold_cross_entropy = threshold_cross_entropy

    def hard_judge(self, infer_result=None):
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
            logger.warning(f'infer result is invalid, value: {infer_result}!')
            return False
        elif len(infer_result) == 0:
            return False
        else:
            log_sum = 0.0
            data_check_list = [class_probability for class_probability
                               in infer_result
                               if data_check(class_probability)]
            if len(data_check_list) == len(infer_result):
                for class_data in data_check_list:
                    log_sum += class_data * math.log(class_data)
                confidence_score = 1 + 1.0 * log_sum / math.log(
                    len(infer_result))
                return confidence_score >= self.threshold_cross_entropy
            else:
                logger.warning("every value of infer_result should be in "
                               f"[0,1], your data is {infer_result}")
                return False

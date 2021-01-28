class BaseFilter:
    """The base class to define unified interface."""

    def hard_judge(self, infer_result=None):
        """predict function, and it must be implemented by
        different methods class.

        :param infer_result: prediction result
        :return: `True` means hard sample, `False` means not a hard sample.
        """
        raise NotImplementedError


class ThresholdFilter(BaseFilter):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def hard_judge(self, infer_result=None):
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

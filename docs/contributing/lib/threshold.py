import abc
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.algorithms.hard_example_mining import BaseFilter

@ClassFactory.register(ClassType.HEM, alias="Threshold")
class ThresholdFilter(BaseFilter, abc.ABC):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = float(threshold)

    def __call__(self, infer_result=None):
        # if invalid input, return False
        if not (infer_result 
                and all(map(lambda x: len(x) > 4, infer_result))):
            return False
            
        image_score = 0
        
        for bbox in infer_result:
            image_score += bbox[4]
            
        average_score = image_score / (len(infer_result) or 1)
        return average_score < self.threshold

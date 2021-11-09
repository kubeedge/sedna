# Algorithm Development Guide

New algorithms, such as `hard example mining` in `incremental_learning` and `joint_inference`, `aggreagtion` in `federated_learning`, `multiple task learning` and `unseen task detect` in `lifelong learning`, need to be extended based on the basic classes provided by Sedna.
## 1. Add an hard example mining algorithm

The algorithm named `Threshold` is used as an example to describe how to add an HEM algorithm to the Sedna hard example mining algorithm library.

### 1.1 Starting from the `class_factory.py`

First, let's start from the `class_factory.py`. Two classes are defined in `class_factory.py`, namely `ClassType` and `ClassFactory`.

`ClassFactory` can register the modules you want to reuse through decorators. For the new `ClassType.HEM` algorithm, the code is as follows:

```python

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

```

## 2. Configuring in the CRD yaml

After registration, you only need to change the name of the hem and parameters in the yaml file, and then the corresponding class will be automatically called according to the name.

```yaml
deploySpec:
    hardExampleMining:
      name: "Threshold"
      parameters:
        - key: "threshold"
          value: "0.9"
```

import numpy as np
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.STP)
class TaskAllocationSimple:
    """
    Corresponding to `TaskDefinitionByOrigin`

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    origins: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, task_extractor, **kwargs):
        self.task_extractor = task_extractor

    def __call__(self, samples: BaseDataSource):
        allocations = np.zeros(samples.num_examples(), dtype=np.int8)
        for (i, data) in enumerate(samples.x): 
            if "garden" in data[0]:
                allocations[i] = 1
       
        return samples, allocations

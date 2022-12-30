import numpy as np
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

from .base_task_allocation import BaseTaskAllocation

@ClassFactory.register(ClassType.STP)
class TaskAllocationSimple(BaseTaskAllocation):
    """
    Corresponding to `TaskDefinitionByOrigin`

    Parameters
    ----------
    task_extractor : Dict
        used to predict target tasks for each inference sample
    origins: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, task_extractor, **kwargs):
        super(TaskAllocationSimple, self).__init__(task_extractor)

    def __call__(self, samples: BaseDataSource):
        allocations = np.zeros(samples.num_examples(), dtype=np.int8)
        for (i, data) in enumerate(samples.x):
            if "garden" in data[0]:
                allocations[i] = 1

        return samples, allocations

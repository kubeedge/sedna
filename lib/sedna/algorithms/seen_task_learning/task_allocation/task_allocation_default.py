import numpy as np
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.STP)
class TaskAllocationDefault:
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
        import numpy.random as rand
        allocations = [rand.randint(0, 2) for _ in range(samples.num_examples())]
       
        return samples, allocations

"""
Mining tasks of inference sample base on task attribute extractor

Parameters
----------
samples ： infer sample, see `sedna.datasources.BaseDataSource` for more detail.

Returns
-------
allocations : tasks that assigned to each sample
"""

from sedna.datasources import BaseDataSource


class BaseTaskAllocation:
    """
    Base class of task allocation algorithm

    Parameters
    ----------
    task_extractor : Model or dict
        task extractor is used to predict target tasks
    """

    def __init__(self, task_extractor, **kwargs):
        self.task_extractor = task_extractor

    def __call__(self, samples: BaseDataSource):
        raise NotImplementedError

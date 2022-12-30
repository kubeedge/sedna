'''
Mining tasks of inference unseen sample 
base on unseen task attribute extractor

Parameters
----------
samples ： infer unseen sample, 
see `sedna.datasources.BaseDataSource` for more detail.

Returns
-------
allocations : tasks that assigned to each sample
'''

from sedna.datasources import BaseDataSource


class BaseUnseenTaskAllocation:
    # TODO: to be completed
    """
    Task allocation for unseen data

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    """

    def __init__(self, task_extractor, **kwargs):
        self.task_extractor = task_extractor

    def __call__(self, samples: BaseDataSource):
        '''
        Parameters
        ----------
        samples: samples to be allocated

        Returns
        -------
        samples: BaseDataSource
            grouped samples based on allocations
        allocations: List
            allocation decision for actual inference
        '''

        raise NotImplementedError


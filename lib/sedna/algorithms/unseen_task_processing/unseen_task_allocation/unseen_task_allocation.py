from sedna.common.log import LOGGER
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('UnseenTaskAllocationDefault', )


@ClassFactory.register(ClassType.UTP)
class UnseenTaskAllocationDefault:
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
        self.log = LOGGER

    def __call__(self, samples: BaseDataSource):
        '''
        Parameters
        ----------
        samples: samples to be allocated

        Returns
        -------
        samples: BaseDataSource
        allocations: List
            allocation decision for actual inference
        '''

        try:
            allocations = [self.task_extractor.fit(
                sample) for sample in samples.x]
        except Exception as err:
            self.log.exception(err)

            allocations = [0] * len(samples)
            self.log.info("Use the first task to inference all the samples.")

        return samples, allocations

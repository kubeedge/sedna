from typing import Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('SampleRegonitionRobotic')

@ClassFactory.register(ClassType.UTD)
class SampleRegonitionRobotic:
    '''
    Divide inference samples into seen samples and unseen samples

    Parameters
    ----------
    task_index: str or dict
        knowledge base index which includes indexes of tasks, samples and etc.
    '''

    def __init__(self, task_index, **kwargs):
        pass

    def __call__(self, samples: BaseDataSource) -> Tuple[BaseDataSource, BaseDataSource]:
        '''
        Parameters
        ----------
        samples : BaseDataSource
            inference samples

        Returns
        -------
        seen_task_samples: BaseDataSource
        unseen_task_samples: BaseDataSource
        '''
        seen_task_samples = BaseDataSource(data_type=samples.data_type)
        seen_task_samples.x = samples.x
        unseen_task_samples = BaseDataSource(data_type=samples.data_type)
        # unseen_task_samples.x = samples.x

        return seen_task_samples, unseen_task_samples, None, None
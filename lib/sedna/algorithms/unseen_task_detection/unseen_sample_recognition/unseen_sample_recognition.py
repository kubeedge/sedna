from typing import Tuple

from sedna.common.file_ops import FileOps
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

from .base_unseen_sample_recognition import BaseSampleRegonition

__all__ = ('SampleRegonitionDefault', 'SampleRegonitionByRFNet')


@ClassFactory.register(ClassType.UTD)
class SampleRegonitionDefault(BaseSampleRegonition):
    '''
    Divide inference samples into seen samples and unseen samples

    Parameters
    ----------
    task_index: str or dict
        knowledge base index which includes indexes of tasks, samples and etc.
    '''

    def __init__(self, task_index, **kwargs):
        super(SampleRegonitionDefault, self).__init__(task_index)

    def __call__(self,
                 samples: BaseDataSource) -> Tuple[BaseDataSource,
                                                   BaseDataSource]:
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
        import random
        seen_task_samples = BaseDataSource(data_type=samples.data_type)
        unseen_task_samples = BaseDataSource(data_type=samples.data_type)

        if samples.num_examples() == 1:
            random_index = random.randint(0, 1)
            if random_index == 0:
                seen_task_samples.x = []
                unseen_task_samples.x = samples.x
            else:
                seen_task_samples.x = samples.x
                unseen_task_samples.x = []
            return seen_task_samples, unseen_task_samples

        sample_num = int(len(samples.x) / 2)
        seen_task_samples.x = samples.x[:sample_num]
        unseen_task_samples.x = samples.x[sample_num:]

        return seen_task_samples, unseen_task_samples

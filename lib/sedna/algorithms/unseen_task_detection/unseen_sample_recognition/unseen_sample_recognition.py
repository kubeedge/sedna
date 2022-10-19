from typing import Tuple

from sedna.common.file_ops import FileOps
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('SampleRegonitionDefault', 'SampleRegonitionByRFNet')


@ClassFactory.register(ClassType.UTD)
class SampleRegonitionDefault:
    '''
    Divide inference samples into seen samples and unseen samples

    Parameters
    ----------
    task_index: str or dict
        knowledge base index which includes indexes of tasks, samples and etc.
    '''

    def __init__(self, task_index, **kwargs):
        if isinstance(task_index, str) and FileOps.exists(task_index):
            self.task_index = FileOps.load(task_index)
        else:
            self.task_index = task_index

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
        unseen_task_samples = BaseDataSource(data_type=samples.data_type)\

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

@ClassFactory.register(ClassType.UTD)
class SampleRegonitionByRFNet:
    '''
    Divide inference samples into seen samples and unseen samples by confidence.

    Parameters
    ----------
    task_index: str or dict
        knowledge base index which includes indexes of tasks, samples and etc.
    '''

    def __init__(self, task_index, **kwargs):
        if isinstance(task_index, str) and FileOps.exists(task_index):
            self.task_index = FileOps.load(task_index)
        else:
            self.task_index = task_index

        self.validator = kwargs.get("validator")

    def __call__(self, samples: BaseDataSource, **
                 kwargs) -> Tuple[BaseDataSource, BaseDataSource]:
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
        from torch.utils.data import DataLoader

        self.validator.test_loader = DataLoader(
            samples.x, batch_size=1, shuffle=False)

        seen_task_samples = BaseDataSource(data_type=samples.data_type)
        unseen_task_samples = BaseDataSource(data_type=samples.data_type)

        seen_task_samples.x, unseen_task_samples.x = self.validator.task_divide()

        return seen_task_samples, unseen_task_samples


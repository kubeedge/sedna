from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('SampleReRegonitionDefault', )


@ClassFactory.register(ClassType.UTD)
class SampleReRegonitionDefault:
    # TODO: to be completed
    '''
    Divide inference samples into seen tasks and unseen tasks.

    Parameters
    ----------
    task_index: str or Dict
    '''

    def __init__(self, task_index, **kwargs):
        pass

    def __call__(self, samples: BaseDataSource):
        '''
        Parameters
        ----------
        samples: training samples

        Returns
        -------
        seen_task_samples: BaseDataSource
        unseen_task_samples: BaseDataSource
        '''

        sample_num = int(len(samples.x) / 2)

        seen_task_samples = BaseDataSource(data_type=samples.data_type)
        seen_task_samples.x = samples.x[:sample_num]
        seen_task_samples.y = samples.y[:sample_num]

        unseen_task_samples = BaseDataSource(data_type=samples.data_type)
        unseen_task_samples.x = samples.x[sample_num:]
        unseen_task_samples.y = samples.y[sample_num:]

        return seen_task_samples, unseen_task_samples

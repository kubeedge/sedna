# Copyright 2023 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

from .base_unseen_sample_re_recognition import BaseSampleReRegonition

__all__ = ('SampleReRegonitionDefault', )


@ClassFactory.register(ClassType.UTD)
class SampleReRegonitionDefault(BaseSampleReRegonition):
    # TODO: to be completed
    '''
    Divide labeled unseen samples into seen tasks and unseen tasks.

    Parameters
    ----------
    task_index: str or Dict
        knowledge base index which includes indexes
        of tasks, samples, models, etc.
    '''

    def __init__(self, task_index, **kwargs):
        super(SampleReRegonitionDefault, self).__init__(task_index)

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

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

from typing import Tuple

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

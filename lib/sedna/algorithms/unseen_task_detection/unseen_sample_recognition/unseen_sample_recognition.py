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
    Treat all inference samples as seen when no recognizer is configured.

    Configure a custom recognizer to identify unseen samples.

    Parameters
    ----------
    task_index: str or dict
        knowledge base index which includes indexes of tasks, samples and etc.
    '''

    def __init__(self, task_index, **kwargs):
        super(SampleRegonitionDefault, self).__init__(task_index)

    def __call__(self,
                 samples: BaseDataSource,
                 **kwargs) -> Tuple[BaseDataSource, BaseDataSource]:
        '''
        Parameters
        ----------
        samples : BaseDataSource
            inference samples
        kwargs : Dict
            extra inference parameters, ignored by this default recognizer

        Returns
        -------
        seen_task_samples: BaseDataSource
            all input samples, in their original order
        unseen_task_samples: BaseDataSource
            empty datasource with the same data type as the input
        '''
        seen_task_samples = BaseDataSource(data_type=samples.data_type)
        unseen_task_samples = BaseDataSource(data_type=samples.data_type)

        seen_task_samples.x = samples.x
        unseen_task_samples.x = []

        return seen_task_samples, unseen_task_samples

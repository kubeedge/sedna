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

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

"""
Divide labeled unseen samples into seen tasks and unseen tasks.

Parameters
----------
task_index: str or Dict
    knowledge base index which includes indexes of
    tasks, samples, models, etc.

Returns
-------
seen_task_samples: seen samples, see `sedna.datasources.BaseDataSource`
for more detail
unseen_task_samples: unseen samples, see `sedna.datasources.BaseDataSource`
for more detail
"""

from sedna.common.file_ops import FileOps
from sedna.datasources import BaseDataSource


class BaseSampleReRegonition:
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
        if isinstance(task_index, str):
            if not FileOps.exists(task_index):
                raise Exception(f"{task_index} not exists!")

            self.task_index = FileOps.load(task_index)
        else:
            self.task_index = task_index

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

        raise NotImplementedError

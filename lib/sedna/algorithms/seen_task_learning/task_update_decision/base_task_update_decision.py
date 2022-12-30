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
Divide multiple tasks based on data

Parameters
----------
samples： Train data, see `sedna.datasources.BaseDataSource` for more detail.

Returns
-------
tasks: All tasks based on training data.
task_extractor: Model or dict with a method to predict target tasks
"""

from typing import List, Dict, Tuple

from sedna.common.file_ops import FileOps
from sedna.common.constant import KBResourceConstant
from sedna.datasources import BaseDataSource

from ..artifact import Task


class BaseTaskUpdateDecision:
    """
    Decide processing strategies for different tasks
    with labeled unseen samples.
    Turn unseen samples to be seen.

    Parameters
    ----------
    task_index: str or Dict
    """

    def __init__(self, task_index, **kwargs):
        if isinstance(task_index, str):
            if not FileOps.exists(task_index):
                raise Exception(f"{task_index} not exists!")
            self.task_index = FileOps.load(task_index)
        else:
            self.task_index = task_index

        self.seen_task_key = KBResourceConstant.SEEN_TASK.value
        self.unseen_task_key = KBResourceConstant.UNSEEN_TASK.value
        self.task_group_key = KBResourceConstant.TASK_GROUPS.value
        self.extractor_key = KBResourceConstant.EXTRACTOR.value

    def __call__(self,
                 samples: BaseDataSource) -> Tuple[List[Task], Dict]:
        raise NotImplementedError

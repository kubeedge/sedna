# Copyright 2021 The KubeEdge Authors.
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
task_extractor: Model with a method to predicting target tasks
"""

import time

from sedna.common.file_ops import FileOps
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.artifact import Task

__all__ = ('UpdateStrategyDefault', )


@ClassFactory.register(ClassType.KM)
class UpdateStrategyDefault:
    """
    Decide processing strategies for different tasks

    Parameters
    ----------
    task_index: str or Dict
    """

    def __init__(self, task_index, **kwargs):
        if isinstance(task_index, str):
            task_index = FileOps.load(task_index)
        self.task_index = task_index

    def __call__(self, samples, task_type):
        """
        Parameters
        ----------
        samples: BaseDataSource
            seen task samples or unseen task samples to be processed.
        task_type: str
            "seen_task" or "unseen_task".
            See sedna.common.constant.KBResourceConstant for more details.

        Returns
        -------
        self.tasks: List[Task]
            tasks to be processed.
        task_update_strategies: Dict
            strategies to process each task.
        """

        if task_type == "seen_task":
            task_index = self.task_index["seen_task"]
        else:
            task_index = self.task_index["unseen_task"]

        self.extractor = task_index["extractor"]
        task_groups = task_index["task_groups"]

        tasks = [task_group.tasks[0] for task_group in task_groups]

        task_update_strategies = {}
        for task in tasks:
            task_update_strategies[task.entry] = {
                "raw_data_update": None,
                "target_model_update": None,
                "task_attr_update": None,
            }

        x_data = samples.x
        y_data = samples.y
        d_type = samples.data_type

        for task in tasks:
            origin = task.meta_attr
            _x = [x for x in x_data if origin in x[0]]
            _y = [y for y in y_data if origin in y]

            task_df = BaseDataSource(data_type=d_type)
            task_df.x = _x
            task_df.y = _y

            task.samples = task_df

            task_update_strategies[task.entry] = {
                "raw_data_update": samples,
                "target_model_update": samples,
                "task_attr_update": samples
            }

        return tasks, task_update_strategies

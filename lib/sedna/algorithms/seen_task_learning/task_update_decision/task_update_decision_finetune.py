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

from typing import List, Tuple, Dict

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory

from ..artifact import Task
from .base_task_update_decision import BaseTaskUpdateDecision

__all__ = ('UpdateStrategyDefault', )


@ClassFactory.register(ClassType.KM)
class UpdateStrategyByFinetune(BaseTaskUpdateDecision):
    """
    Finetuning all the samples based on knowledgebase
    to turn unseen samples to be seen.

    Parameters
    ----------
    task_index: str or Dict
    """

    def __init__(self, task_index, **kwargs):
        super(UpdateStrategyByFinetune, self).__init__(task_index)
        self.attribute = kwargs.get("attribute").split(", ")
        if not self.attribute:
            self.attribute = ("real", "sim")
        self.city = kwargs.get("city", "berlin")

    def __call__(self, samples, task_type) -> Tuple[List[Task], Dict]:
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

        if task_type == self.seen_task_key:
            task_index = self.task_index[self.seen_task_key]
        else:
            task_index = self.task_index[self.unseen_task_key]

        task_groups = task_index[self.task_group_key]
        tasks = [task_group.tasks[0] for task_group in task_groups]

        d_type = samples.data_type
        sample_index = range(samples.num_examples())

        task_update_strategies = {}
        for task in tasks:
            task_update_strategies[task.entry] = {
                "raw_data_update": None,
                "target_model_update": None,
                "task_attr_update": None,
            }

        _idx = [i for i in sample_index if self.city in samples.y[i]]
        _y = samples.y[_idx]
        _x = samples.x[_idx]
        _sample = BaseDataSource(data_type=d_type)
        _sample.x, _sample.y = _x, _y
        task = tasks[[i for i, task in enumerate(tasks) if
                      task.meta_attr == self.attribute[0]]][0]
        task.samples = _sample
        task_update_strategies[task.entry] = {
            "raw_data_update": _sample,
            "target_model_update": True,
            "task_attr_update": None,
        }

        _idx = list(set(sample_index) - set(_idx))
        _y = samples.y[_idx]
        _x = samples.x[_idx]
        _sample = BaseDataSource(data_type=d_type)
        _sample.x, _sample.y = _x, _y
        task = tasks[[i for i, task in enumerate(
            tasks) if task.meta_attr == self.attribute[1]]][0]
        task.samples = _sample
        task_update_strategies[task.entry] = {
            "raw_data_update": _sample,
            "target_model_update": True,
            "task_attr_update": None,
        }
        return tasks, task_update_strategies

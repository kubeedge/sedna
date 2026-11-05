# Copyright 2026 The KubeEdge Authors.
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

"""Default update decision for existing lifelong-learning tasks."""

from typing import Dict, List, Tuple

from sedna.common.class_factory import ClassFactory, ClassType
from sedna.datasources import BaseDataSource

from ..artifact import Task
from .base_task_update_decision import BaseTaskUpdateDecision

__all__ = ('UpdateStrategyDefault', )


@ClassFactory.register(ClassType.KM)
class UpdateStrategyDefault(BaseTaskUpdateDecision):
    """Update every existing task with all the supplied labeled samples.

    The task index can be a dictionary or a path to a saved index. Existing
    models and task attributes are retained so the learner can load and
    retrain each model using the new samples, without task-specific parameters.
    """

    def __call__(self, samples: BaseDataSource,
                 task_type: str) -> Tuple[List[Task], Dict]:
        """Return tasks and update strategies for seen or unseen task groups.

        ``task_type`` selects ``seen_task`` or ``unseen_task`` in the index.
        Each task receives the supplied samples in place of its old samples;
        its model is marked for updating and its attributes are left unchanged.
        Empty task groups produce no tasks or strategies.
        """
        task_groups = self.task_index[task_type][self.task_group_key]
        tasks = [task for group in task_groups for task in group.tasks]
        task_update_strategies = {}
        for task in tasks:
            task.samples = samples
            task_update_strategies[task.entry] = {
                "raw_data_update": samples,
                "target_model_update": True,
                "task_attr_update": None,
            }
        return tasks, task_update_strategies

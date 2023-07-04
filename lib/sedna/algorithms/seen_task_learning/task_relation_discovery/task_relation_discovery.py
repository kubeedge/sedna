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
Discover relationships between all tasks

Parameters
----------
tasks ：all tasks form `task_definition`

Returns
-------
task_groups : List of groups which including at least 1 task.
"""

from typing import List

from sedna.common.class_factory import ClassType, ClassFactory

from ..artifact import Task, TaskGroup
from .base_task_relation_discovery import BaseTaskRelationDiscover


__all__ = ('DefaultTaskRelationDiscover',)


@ClassFactory.register(ClassType.STP)
class DefaultTaskRelationDiscover(BaseTaskRelationDiscover):
    """
    Assume that each task is independent of each other
    """

    def __init__(self, **kwargs):
        super(DefaultTaskRelationDiscover, self).__init__(**kwargs)

    def __call__(self, tasks: List[Task]) -> List[TaskGroup]:
        tgs = []
        for task in tasks:
            tg_obj = TaskGroup(entry=task.entry, tasks=[task])
            tg_obj.samples = task.samples
            tgs.append(tg_obj)
        return tgs

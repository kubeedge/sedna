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

from typing import List
from .artifact import Task, TaskGroup
from sedna.common.class_factory import ClassType, ClassFactory


__all__ = ('DefaultTaskRelationDiscover', )


@ClassFactory.register(ClassType.MTL)
class DefaultTaskRelationDiscover:
    def __init__(self, **kwargs):
        pass

    def __call__(self, tasks: List[Task]) -> List[TaskGroup]:
        tg = []
        for task in tasks:
            tg_obj = TaskGroup(entry=task.entry, tasks=[task])
            tg_obj.samples = task.samples
            tg.append(tg_obj)
        return tg

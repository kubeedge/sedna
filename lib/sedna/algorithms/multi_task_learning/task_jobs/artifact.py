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

__all__ = ('Task', 'TaskGroup', 'Model')


class Task:
    def __init__(self, entry, samples, meta_attr=None):
        """
        task instance
        :param entry: task name, unique, e.g. Shenzhen_Spring
        :param samples: datasource
        :param meta_attr: meta data use to describe task
        """
        self.entry = entry
        self.samples = samples
        self.meta_attr = meta_attr
        self.test_samples = None  # assign on task definition and use in TRD
        self.model = None  # assign on running
        self.result = None  # assign on running


class TaskGroup:
    def __init__(self, entry, tasks: List[Task]):
        """
        task group instance, set of tasks
        :param entry: task group name
        :param tasks: set of tasks
        """
        self.entry = entry
        self.tasks = tasks
        self.samples = None  # assign with task_relation_discover algorithms
        self.model = None  # assign on train


class Model:
    def __init__(self, index: int, entry, model, result):
        """
        model instance
        :param index: kb index, e.g. 1
        :param entry: model name
        :param model: model save path
        :param result: evaluation result
        """
        self.index = index  # integer
        self.entry = entry
        self.model = model
        self.result = result
        self.meta_attr = None  # assign on running

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
Remodeling tasks based on their relationships

Parameters
----------
mappings ：all assigned tasks get from the `task_allocation`
samples : input samples

Returns
-------
models : List of groups which including at least 1 task.
"""

from typing import List

from sedna.datasources import BaseDataSource


class BaseTaskRemodeling:
    """
    Assume that each task is independent of each other
    """

    def __init__(self, models: list, **kwargs):
        self.models = models

    def __call__(self, samples: BaseDataSource, mappings: List):
        """
        Grouping based on assigned tasks
        """
        raise NotImplementedError

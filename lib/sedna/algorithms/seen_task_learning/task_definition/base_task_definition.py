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

from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource

from ..artifact import Task


class BaseTaskDefinition:
    """
    Dividing datasets with all sorts of methods
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self,
                 samples: BaseDataSource) -> Tuple[List[Task],
                                                   Any,
                                                   BaseDataSource]:
        raise NotImplementedError

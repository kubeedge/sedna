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

from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory

from ..artifact import Task
from .base_task_definition import BaseTaskDefinition


@ClassFactory.register(ClassType.STP)
class TaskDefinitionByOrigin(BaseTaskDefinition):
    """
    Dividing datasets based on the their origins.

    Parameters
    ----------
    attr_filed Tuple[Metadata]
        metadata is usually a class feature label with a finite values.
    """

    def __init__(self, **kwargs):
        super(TaskDefinitionByOrigin, self).__init__()
        self.attribute = kwargs.get("attribute").split(", ")
        self.city = kwargs.get("city")

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:

        tasks = []
        d_type = samples.data_type

        task_index = dict(zip(self.attribute, range(len(self.attribute))))
        sample_index = range(samples.num_examples())

        _idx = [i for i in sample_index if self.city in samples.y[i]]
        _y = samples.y[_idx]
        _x = samples.x[_idx]
        _sample = BaseDataSource(data_type=d_type)
        _sample.x, _sample.y = _x, _y

        g_attr = f"{self.attribute[0]}"
        task_obj = Task(entry=g_attr, samples=_sample,
                        meta_attr=self.attribute[0])
        tasks.append(task_obj)

        _idx = list(set(sample_index) - set(_idx))
        _y = samples.y[_idx]
        _x = samples.x[_idx]
        _sample = BaseDataSource(data_type=d_type)
        _sample.x, _sample.y = _x, _y

        g_attr = f"{self.attribute[-1]}"
        task_obj = Task(entry=g_attr, samples=_sample,
                        meta_attr=self.attribute[-1])
        tasks.append(task_obj)

        return tasks, task_index, samples

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

import numpy as np
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

from .base_task_allocation import BaseTaskAllocation

# TODO: this class is just for demonstrate


@ClassFactory.register(ClassType.STP)
class TaskAllocationStream(BaseTaskAllocation):
    """
    Corresponding to `TaskDefinitionByOrigin`

    Parameters
    ----------
    task_extractor : Dict
        used to predict target tasks for each inference sample
    origins: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, task_extractor, **kwargs):
        super(TaskAllocationStream, self).__init__(task_extractor)

    def __call__(self, samples: BaseDataSource):
        allocations = [np.random.randint(0, 1)
                       for _ in range(samples.num_examples())]

        return samples, allocations

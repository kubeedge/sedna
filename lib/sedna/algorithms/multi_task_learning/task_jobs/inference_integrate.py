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

import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType

from .artifact import Task

__all__ = ('DefaultInferenceIntegrate', )


@ClassFactory.register(ClassType.MTL)
class DefaultInferenceIntegrate:
    """
    make the largest results in each model as the final prediction result
    generally used for regression prediction
    """
    def __init__(self, models: list, **kwargs):
        self.models = models

    def __call__(self, tasks: List[Task]):
        res = {}
        for task in tasks:
            res.update(dict(zip(task.samples.inx, task.result)))
        return np.array([z[1]
                        for z in sorted(res.items(), key=lambda x: x[0])])

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
Integrate the inference results of all related tasks
"""

from typing import List

import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType

from .base_inference_integrate import BaseInferenceIntegrate
from ..artifact import Task

__all__ = ('DefaultInferenceIntegrate', )


@ClassFactory.register(ClassType.STP)
class DefaultInferenceIntegrate(BaseInferenceIntegrate):
    """
    Default calculation algorithm for inference integration

    Parameters
    ----------
    models: All models used for sample inference
    """

    def __init__(self, models: list, **kwargs):
        super(DefaultInferenceIntegrate, self).__init__(models)

    def __call__(self, tasks: List[Task]):
        """
        Parameters
        ----------
        tasks: All tasks with sample result

        Returns
        -------
        result: minimum result
        """
        res = {}
        for task in tasks:
            res.update(dict(zip(task.samples.inx, task.result)))
        return np.array([z[1]
                        for z in sorted(res.items(), key=lambda x: x[0])])

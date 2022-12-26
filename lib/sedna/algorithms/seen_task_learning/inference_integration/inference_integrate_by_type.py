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

"""
Integrate the inference results of all related tasks
"""

from typing import List

import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType

from ..artifact import Task

__all__ = ('InferenceIntegrateByType', )


@ClassFactory.register(ClassType.STP)
class InferenceIntegrateByType:
    """
    Default calculation algorithm for inference integration

    Parameters
    ----------
    models: All models used for sample inference
    """

    def __init__(self, models: list, **kwargs):
        self.models = models

    def __call__(self, tasks: List[Task]):
        """
        Parameters
        ----------
        tasks: All tasks with sample result

        Returns
        -------
        result: minimum result
        """

        curb_results, ramp_results = [], []
        for task in tasks:
            curb_result, ramp_result = task.result
            curb_results = curb_result if not curb_results else np.concatenate((curb_results, curb_result))
            ramp_results = ramp_result if not ramp_results else np.concatenate((ramp_results, ramp_result))

        return curb_results, ramp_results


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

from ..artifact import Task


class BaseInferenceIntegrate:
    """
    Base class for default calculation algorithm for inference integration

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
        result: inference results for all the inference samples
        """
        raise NotImplementedError

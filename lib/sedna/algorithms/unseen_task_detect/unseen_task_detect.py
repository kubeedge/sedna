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
Unseen task detection algorithms for Lifelong Learning
"""

import abc
from typing import List

import numpy as np

from sedna.algorithms.multi_task_learning.task_jobs.artifact import Task
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('ModelProbeFilter', 'TaskAttrFilter')


class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define unified interface."""

    def __call__(self, tasks: Task = None):
        """
        predict function, and it must be implemented by
        different methods class.

        Parameters
        ----------
        tasks : inference task

        Returns
        -------
        is unseen task : bool
            `True` means unseen task, `False` means not.
        """
        raise NotImplementedError


@ClassFactory.register(ClassType.UTD)
class ModelProbeFilter(BaseFilter, abc.ABC):
    """
    Judgment based on the confidence of the prediction result,
    typically used for classification problems
    """

    def __init__(self):
        pass

    def __call__(self, tasks: List[Task] = None, threshold=0.5, **kwargs):
        """
        Parameters
        ----------
        tasks : inference task
        threshold : float
            threshold considered credible

        Returns
        -------
        is unseen task: bool
            `True` means unseen task, `False` means not.
        """
        all_proba = []
        for task in tasks:
            sample = task.samples
            model = task.model
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sample)
                all_proba.append(np.max(proba))
        return np.mean(all_proba) > threshold if all_proba else True


@ClassFactory.register(ClassType.UTD)
class TaskAttrFilter(BaseFilter, abc.ABC):
    """
    Judgment based on whether the metadata of the sample has been found in KB
    """
    def __init__(self):
        pass

    def __call__(self, tasks: List[Task] = None, **kwargs):
        """
        Parameters
        ----------
        tasks : inference task

        Returns
        -------
        is unseen task: bool
            `True` means unseen task, `False` means not.
        """
        for task in tasks:
            model_attr = list(map(list, task.model.meta_attr))
            sample_attr = list(map(list, task.samples.meta_attr))

            if not (model_attr and sample_attr):
                continue
            if list(model_attr) == list(sample_attr):
                return False
        return True

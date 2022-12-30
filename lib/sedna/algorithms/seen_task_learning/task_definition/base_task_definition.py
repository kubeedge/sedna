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
        raise NotImplementedError


    def __call__(self,
                 samples: BaseDataSource) -> Tuple[List[Task],
                                                   Any,
                                                   BaseDataSource]:
        raise NotImplementedError

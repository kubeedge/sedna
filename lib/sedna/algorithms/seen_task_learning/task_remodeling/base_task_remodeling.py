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


"""
Discover relationships between all tasks

Parameters
----------
tasks ：all tasks form `task_definition`

Returns
-------
task_groups : List of groups which including at least 1 task.
"""

from typing import List

from ..artifact import Task, TaskGroup


class BaseTaskRelationDiscover:
    """
    Assume that each task is independent of each other
    """

    def __init__(self, **kwargs):
        raise NotImplementedError

    def __call__(self, tasks: List[Task]) -> List[TaskGroup]:
        raise NotImplementedError


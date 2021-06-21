Module sedna.algorithms.multi_task_learning.task_jobs.artifact
==============================================================

Classes
-------

`Model(index: int, entry, model, result)`
:   Model instance
    :param index: index, ID, e.g. 1
    :param entry: model name, e.g. Shenzhen_Spring
    :param model: ai model save path
    :param result: evaluation result

`Task(entry: str, samples, meta_attr=None)`
:   task instance
    :param entry: task name, unique, e.g. Shenzhen_Spring
    :param samples: datasource
    :param meta_attr: meta data describe task

`TaskGroup(entry: str, tasks: List[sedna.algorithms.multi_task_learning.task_jobs.artifact.Task])`
:   task_group instance
    :param entry: task group name, unique, e.g. Shenzhen_Spring
    :param tasks: set of Task
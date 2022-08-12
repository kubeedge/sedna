from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory

from ..artifact import Task

@ClassFactory.register(ClassType.STP)
class TaskDefinitionByOrigin:
    """
    Dividing datasets based on the their origins.

    Parameters
    ----------
    attribute： List[Metadata]
        metadata is usually a class feature label with a finite values.
    """

    def __init__(self, **kwargs):
        self.origins = kwargs.get("origins", [])

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:
        tasks = []
        d_type = samples.data_type
        x_data = samples.x
        y_data = samples.y

        task_index = dict(zip(self.origins, range(len(self.origins))))

        for k, v in task_index.items():
            _x = [x for x in x_data if k in x[0]]
            _y = [y for y in y_data if k in y]

            task_df = BaseDataSource(data_type=d_type)
            task_df.x = _x
            task_df.y = _y

            g_attr = f"{k}_semantic_segamentation_model"
            task_obj = Task(entry=g_attr, samples=task_df, meta_attr=k)
            tasks.append(task_obj)

        samples = BaseDataSource(data_type=d_type)
        samples.x = x_data
        samples.y = y_data

        return tasks, task_index, samples
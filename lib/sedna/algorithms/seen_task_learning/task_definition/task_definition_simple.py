from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory

from ..artifact import Task

@ClassFactory.register(ClassType.STP)
class TaskDefinitionSimple:
    """
    Dividing datasets based on the their origins.

    Parameters
    ----------
    origins： List[Metadata]
        metadata is usually a class feature label with a finite values.
    """

    def __init__(self, **kwargs):
        self.origins = ["front", "garden"]

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:

        tasks = []
        d_type = samples.data_type
        x_data = samples.x
        y_data = samples.y

        task_index = dict(zip(self.origins, range(len(self.origins))))

        front_data = BaseDataSource(data_type=d_type)
        front_data.x, front_data.y = [], []
        garden_data = BaseDataSource(data_type=d_type)
        garden_data.x, garden_data.y = [], []

        for (i, data) in enumerate(x_data):
            if "front" in data[0]:
                front_data.x.append(data)
                front_data.y.append(y_data[i])
            else:
                garden_data.x.append(data)
                garden_data.y.append(y_data[i])

        g_attr_front = "front_semantic_segamentation_model"
        front_task = Task(entry=g_attr_front, samples=front_data, meta_attr="front")
        tasks.append(front_task)

        g_attr_garden = "garden_semantic_segamentation_model"
        garden_task = Task(entry=g_attr_garden, samples=garden_data, meta_attr="garden")
        tasks.append(garden_task)

        return tasks, task_index, samples

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
Divide multiple tasks based on data

Parameters
----------
samples： Train data, see `sedna.datasources.BaseDataSource` for more detail.

Returns
-------
tasks: All tasks based on training data.
task_extractor: Model with a method to predicting target tasks
"""

from typing import List, Any, Tuple

import numpy as np
import pandas as pd

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory

from .artifact import Task


__all__ = ('TaskDefinitionBySVC', 'TaskDefinitionByDataAttr')


@ClassFactory.register(ClassType.MTL)
class TaskDefinitionBySVC:
    """
    Dividing datasets with `AgglomerativeClustering` based on kernel distance,
    Using SVC to fit the clustering result.

    Parameters
    ----------
    n_class： int or None
        The number of clusters to find, default=2.
    """

    def __init__(self, **kwargs):
        n_class = kwargs.get("n_class", "")
        self.n_class = max(2, int(n_class)) if str(n_class).isdigit() else 2

    def __call__(self,
                 samples: BaseDataSource) -> Tuple[List[Task],
                                                   Any,
                                                   BaseDataSource]:
        from sklearn.svm import SVC
        from sklearn.cluster import AgglomerativeClustering

        d_type = samples.data_type
        x_data = samples.x
        y_data = samples.y
        if not isinstance(x_data, pd.DataFrame):
            raise TypeError(f"{d_type} data should only be pd.DataFrame")
        tasks = []
        legal = list(
            filter(lambda col: x_data[col].dtype == 'float64', x_data.columns))

        df = x_data[legal]
        c1 = AgglomerativeClustering(n_clusters=self.n_class).fit_predict(df)
        c2 = SVC(gamma=0.01)
        c2.fit(df, c1)

        for task in range(self.n_class):
            g_attr = f"svc_{task}"
            task_df = BaseDataSource(data_type=d_type)
            task_df.x = x_data.iloc[np.where(c1 == task)]
            task_df.y = y_data.iloc[np.where(c1 == task)]

            task_obj = Task(entry=g_attr, samples=task_df)
            tasks.append(task_obj)
        samples.x = df
        return tasks, c2, samples


@ClassFactory.register(ClassType.MTL)
class TaskDefinitionByDataAttr:
    """
    Dividing datasets based on the common attributes,
    generally used for structured data.

    Parameters
    ----------
    attribute： List[Metadata]
        metadata is usually a class feature label with a finite values.
    """
    def __init__(self, **kwargs):
        self.attr_filed = kwargs.get("attribute", [])

    def __call__(self,
                 samples: BaseDataSource) -> Tuple[List[Task],
                                                   Any,
                                                   BaseDataSource]:
        tasks = []
        d_type = samples.data_type
        x_data = samples.x
        y_data = samples.y
        if not isinstance(x_data, pd.DataFrame):
            raise TypeError(f"{d_type} data should only be pd.DataFrame")

        _inx = 0
        task_index = {}
        for meta_attr, df in x_data.groupby(self.attr_filed):
            if isinstance(meta_attr, (list, tuple, set)):
                g_attr = "_".join(
                    map(lambda x: str(x).replace("_", "-"), meta_attr))
                meta_attr = list(meta_attr)
            else:
                g_attr = str(meta_attr).replace("_", "-")
                meta_attr = [meta_attr]
            g_attr = g_attr.replace(" ", "")
            if g_attr in task_index:
                old_task = tasks[task_index[g_attr]]
                old_task.x = pd.concat([old_task.x, df])
                old_task.y = pd.concat([old_task.y, y_data.iloc[df.index]])
                continue
            task_index[g_attr] = _inx

            task_df = BaseDataSource(data_type=d_type)
            task_df.x = df.drop(self.attr_filed, axis=1)
            task_df.y = y_data.iloc[df.index]

            task_obj = Task(entry=g_attr, samples=task_df, meta_attr=meta_attr)
            tasks.append(task_obj)
            _inx += 1
        x_data.drop(self.attr_filed, axis=1, inplace=True)
        samples = BaseDataSource(data_type=d_type)
        samples.x = x_data
        samples.y = y_data
        return tasks, task_index, samples

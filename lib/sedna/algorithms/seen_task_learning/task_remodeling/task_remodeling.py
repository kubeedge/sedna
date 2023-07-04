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
Remodeling tasks based on their relationships

Parameters
----------
mappings ：all assigned tasks get from the `task_mining`
samples : input samples

Returns
-------
models : List of groups which including at least 1 task.
"""

from typing import List

import numpy as np
import pandas as pd

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

from .base_task_remodeling import BaseTaskRemodeling

__all__ = ('DefaultTaskRemodeling',)


@ClassFactory.register(ClassType.STP)
class DefaultTaskRemodeling(BaseTaskRemodeling):
    """
    Assume that each task is independent of each other
    """

    def __init__(self, models: list, **kwargs):
        super(DefaultTaskRemodeling, self).__init__(models)

    def __call__(self, samples: BaseDataSource, mappings: List):
        """
        Grouping based on assigned tasks
        """
        mappings = np.array(mappings)
        data, models = [], []
        d_type = samples.data_type
        for m in np.unique(mappings):
            task_df = BaseDataSource(data_type=d_type)
            _inx = np.where(mappings == m)
            if isinstance(samples.x, pd.DataFrame):
                task_df.x = samples.x.iloc[_inx]
            else:
                task_df.x = np.array(samples.x)[_inx]
            if d_type != "test":
                if isinstance(samples.x, pd.DataFrame):
                    task_df.y = samples.y.iloc[_inx]
                else:
                    task_df.y = np.array(samples.y)[_inx]
            task_df.inx = _inx[0].tolist()
            if samples.meta_attr is not None:
                task_df.meta_attr = np.array(samples.meta_attr)[_inx]
            data.append(task_df)
            # TODO: if m is out of index
            try:
                model = self.models[m]
            except Exception as err:
                print(f"self.models[{m}] not exists. {err}")
                model = self.models[0]
            models.append(model)
        return data, models

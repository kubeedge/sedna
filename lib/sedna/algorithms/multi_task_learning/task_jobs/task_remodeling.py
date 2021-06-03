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

import numpy as np
from typing import List, Tuple
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('DefaultTaskRemodeling',)


@ClassFactory.register(ClassType.MTL)
class DefaultTaskRemodeling:
    def __init__(self, models: list, **kwargs):
        self.models = models

    def __call__(self, samples: BaseDataSource, mappings: List) \
            -> Tuple[List[BaseDataSource], List]:
        mappings = np.array(mappings)
        data, models = [], []
        d_type = samples.data_type
        for m in np.unique(mappings):
            task_df = BaseDataSource(data_type=d_type)
            _inx = np.where(mappings == m)
            task_df.x = samples.x.iloc[_inx]
            if d_type != "test":
                task_df.y = samples.y.iloc[_inx]
            task_df.inx = _inx[0].tolist()
            task_df.meta_attr = samples.meta_attr.iloc[_inx].values
            data.append(task_df)
            model = self.models[m] or self.models[0]
            models.append(model)
        return data, models

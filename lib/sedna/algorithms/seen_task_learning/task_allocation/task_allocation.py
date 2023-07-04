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
Mining tasks of inference sample based on task attribute extractor

Parameters
----------
samples ： infer sample, see `sedna.datasources.BaseDataSource` for more detail.

Returns
-------
allocations : tasks that assigned to each sample
"""

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType


__all__ = (
    'TaskAllocationBySVC',
    'TaskAllocationByDataAttr',
    'TaskAllocationDefault',
)


@ClassFactory.register(ClassType.STP)
class TaskAllocationBySVC:
    """
    Corresponding to `TaskDefinitionBySVC`

    Parameters
    ----------
    task_extractor : Model
        SVC Model used to predicting target tasks
    """

    def __init__(self, task_extractor, **kwargs):
        self.task_extractor = task_extractor

    def __call__(self, samples: BaseDataSource):
        df = samples.x
        allocations = [0, ] * len(df)
        legal = list(
            filter(lambda col: df[col].dtype == 'float64', df.columns))
        if not len(legal):
            return allocations

        allocations = list(self.task_extractor.predict(df[legal]))
        return samples, allocations


@ClassFactory.register(ClassType.STP)
class TaskAllocationByDataAttr:
    """
    Corresponding to `TaskDefinitionByDataAttr`

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    attr_filed: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, task_extractor, **kwargs):
        self.task_extractor = task_extractor
        self.attr_filed = kwargs.get("attribute", [])

    def __call__(self, samples: BaseDataSource):
        df = samples.x
        meta_attr = df[self.attr_filed]

        allocations = meta_attr.apply(
            lambda x: self.task_extractor.get(
                "_".join(
                    map(lambda y: str(x[y]).replace("_", "-").replace(" ", ""),
                        self.attr_filed)
                ),
                0),
            axis=1).values.tolist()
        samples.x = df.drop(self.attr_filed, axis=1)
        samples.meta_attr = meta_attr
        return samples, allocations


@ClassFactory.register(ClassType.STP)
class TaskAllocationDefault:
    """
    Task allocation specifically for unstructured data

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    """

    def __init__(self, task_extractor, **kwargs):
        self.task_extractor = task_extractor

    def __call__(self, samples: BaseDataSource):
        allocations = [0] * len(samples)

        return samples, allocations

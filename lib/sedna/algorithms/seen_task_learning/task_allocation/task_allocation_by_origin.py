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

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

from .base_task_allocation import BaseTaskAllocation


@ClassFactory.register(ClassType.STP)
class TaskAllocationByOrigin(BaseTaskAllocation):
    """
    Corresponding to `TaskDefinitionByOrigin`

    Parameters
    ----------
    task_extractor : Dict
        used to predict target tasks for each inference samples
    origins: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, task_extractor, **kwargs):
        super(TaskAllocationByOrigin, self).__init__(task_extractor)
        self.default_origin = kwargs.get("default", None)

    def __call__(self, samples: BaseDataSource):
        if self.default_origin:
            return samples, [int(self.task_extractor.get(
                self.default_origin))] * len(samples.x)

        cities = [
            "aachen",
            "berlin",
            "bochum",
            "bremen",
            "cologne",
            "darmstadt",
            "dusseldorf",
            "erfurt",
            "hamburg",
            "hanover",
            "jena",
            "krefeld",
            "monchengladbach",
            "strasbourg",
            "stuttgart",
            "tubingen",
            "ulm",
            "weimar",
            "zurich"]

        sample_origins = []
        for _x in samples.x:
            is_real = False
            for city in cities:
                if city in _x[0]:
                    is_real = True
                    sample_origins.append("real")
                    break
            if not is_real:
                sample_origins.append("sim")

        allocations = [int(self.task_extractor.get(sample_origin))
                       for sample_origin in sample_origins]

        return samples, allocations

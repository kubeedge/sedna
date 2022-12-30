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

from sedna.common.log import LOGGER
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

from .base_unseen_task_allocation import BaseUnseenTaskAllocation

__all__ = ('UnseenTaskAllocationDefault', )


@ClassFactory.register(ClassType.UTP)
class UnseenTaskAllocationDefault(BaseUnseenTaskAllocation):
    # TODO: to be completed
    """
    Task allocation for unseen data

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    """

    def __init__(self, task_extractor, **kwargs):
        super(UnseenTaskAllocationDefault, self).__init__(task_extractor)
        self.log = LOGGER

    def __call__(self, samples: BaseDataSource):
        '''
        Parameters
        ----------
        samples: samples to be allocated

        Returns
        -------
        samples: BaseDataSource
        allocations: List
            allocation decision for actual inference
        '''

        try:
            allocations = [self.task_extractor.fit(
                sample) for sample in samples.x]
        except Exception as err:
            self.log.exception(err)

            allocations = [0] * len(samples)
            self.log.info("Use the first task to inference all the samples.")

        return samples, allocations

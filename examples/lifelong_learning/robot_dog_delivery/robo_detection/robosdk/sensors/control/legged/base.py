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
from typing import Union

from robosdk.utils.util import Config
from robosdk.utils.constant import GaitType
from robosdk.sensors.base import SensorBase


__all__ = ("LeggedController", )



class LeggedController(SensorBase):  # noqa
    def __init__(self, name, config: Config = None):
        super(LeggedController, self).__init__(name=name, config=config)

    def get_curr_gait(self) -> GaitType:
        raise NotImplementedError

    def change_gait(self, gait: Union[str, GaitType]):
        raise NotImplementedError

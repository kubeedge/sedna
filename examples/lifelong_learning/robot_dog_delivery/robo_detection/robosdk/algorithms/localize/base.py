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
import abc
import threading

from robosdk.utils.logger import logging
from robosdk.utils.schema import BasePose


__all__ = ("Localize", )


class Localize(metaclass=abc.ABCMeta):

    def __init__(self, name: str = "base_localizer"):
        self.localize_name = name
        self.state_lock = threading.RLock()
        self.curr_state = BasePose()
        self.logger = logging.bind(instance=self.localize_name, system=True)

    @abc.abstractmethod
    def get_curr_state(self) -> BasePose:
        ...

    @abc.abstractmethod
    def set_curr_state(self, state: BasePose):
        ...

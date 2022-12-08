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

from robosdk.utils.schema import BasePose
from robosdk.utils.schema import PathNode
from robosdk.utils.logger import logging
from robosdk.utils.util import Config
from robosdk.cloud_robotics.task_agent.base import RoboActionHandle

__all__ = ("Navigation", )


class Navigation(metaclass=abc.ABCMeta):

    def __init__(self, name: str, config: Config):
        self.navigation_name = name
        self.config = config
        self.move_base_as = None
        self.goal_lock = threading.RLock()
        self.logger = logging.bind(instance=self.navigation_name, system=True)

    @abc.abstractmethod
    def execute_track(self, plan: PathNode):
        ...

    @abc.abstractmethod
    def goto(self, goal: BasePose):
        ...

    @abc.abstractmethod
    def goto_absolute(self, goal: BasePose):
        ...

    def stop(self):
        self.set_vel(forward=0.0, turn=0.0, execution=3)

    @abc.abstractmethod
    def set_vel(self,
                forward: float = 0,
                turn: float = 0,
                execution: int = 1):
        """
        set velocity to robot
       :param forward: linear velocity in m/s
       :param turn: rotational velocity in m/s
       :param execution: execution time in seconds
       """
        ...

    def set_action_server(self, action: RoboActionHandle):
        self.move_base_as = action

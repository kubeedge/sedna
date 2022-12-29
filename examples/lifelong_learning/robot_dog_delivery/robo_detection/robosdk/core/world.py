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
import os
from typing import Optional
from importlib import import_module

from robosdk.utils.logger import logging
from robosdk.utils.fileops import FileOps
from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory
from robosdk.core.robot import Robot
from robosdk.algorithms.path_planning.base import GridMap


class World:
    """
    Word instance defines the environment which the robot launch, such as map.
    """

    def __init__(self, name: str,
                 service: str = "RosActionService",
                 data_collect: str = "",
                 simulator: Optional = None):
        self.world_name = name
        self.simulator = simulator
        self.logger = logging.bind(instance=self.world_name)
        self.robots = []
        self.world_map = None
        self.service = None
        try:
            _ = import_module("robosdk.cloud_robotics.task_agent")
            driver_cls = ClassFactory.get_cls(ClassType.GENERAL, service)
            self.service = driver_cls(name=self.world_name)
        except Exception as e:
            self.logger.error(f"Initial service {self.world_name} failure, {e}")
        self.collector = None
        if data_collect:
            try:
                _ = import_module("robosdk.cloud_robotics.data_collection")
                driver_cls = ClassFactory.get_cls(
                    ClassType.GENERAL, data_collect)
                self.collector = driver_cls()
            except Exception as e:
                self.logger.error(f"Initial data collection failure, {e}")

    def add_robot(self, robot: Robot):
        if not robot.has_connect:
            robot.connect()
        self.robots.append(robot)

    def load_gridmap(self, map_file: str, panoptic: str = None):
        """
        Initial word map by loading a 2D map with panoptic datas
        :param map_file: map path, file
        :param panoptic: semantic information, yaml
        """
        map_file = FileOps.download(map_file, untar=True)
        if os.path.isdir(map_file):
            map_file = ""
            for p in os.listdir(map_file):
                if p.endswith(".pgm"):
                    map_file = os.path.join(map_file, p)
                    break
        self.world_map = GridMap()
        self.world_map.read_from_pgm(map_file)
        if panoptic:
            panoptic = FileOps.download(panoptic)
            self.world_map.parse_panoptic(panoptic)
        self.world_map.calc_obstacle_map()

    def run(self):
        if self.service:
            self.service.initial_executors(self.robots)
            self.service.start()
            self.service.run()
        if self.collector:
            self.collector.connect()
            self.collector.start()

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
import asyncio
from importlib import import_module

from robosdk.utils.logger import logging
from robosdk.utils.fileops import FileOps
from robosdk.utils.util import Config
from robosdk.utils.context import BaseConfig
from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory
from robosdk.utils.exceptions import SensorError
from robosdk.sensors.base import SensorManage

__all__ = ("_init_cfg", "Robot",)


def _init_cfg(config, kind="robots"):
    if config and config.endswith((".yaml", ".yml")):
        config = FileOps.download(config)
    else:
        config = os.path.join(
            BaseConfig.configPath, kind, f"{config}.yaml"
        )
    return config if os.path.isfile(config) else None


class Robot:
    """
     This class builds robot specific objects by reading
     a configuration and instantiating the necessary robot
     module objects.
    """

    def __init__(self, name: str, config: str = None):
        self.robot_name = name
        cfg = _init_cfg(config=config, kind="robots")
        self.config = Config(cfg)
        self.logger = logging.bind(instance=self.robot_name)
        self.control = None
        self.all_sensors = {}
        self.has_connect = False

    def connect(self):
        if self.config.environment.backend == "ros":
            import rospy
            rospy.init_node(self.robot_name)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.gather(
            self.initial_sensors(),
            self.initial_navigation(),
            self.initial_control(),
        ))
        self.has_connect = True

    def add_sensor(self, sensor: str, name: str, config: Config):
        try:
            _ = import_module(f"robosdk.sensors.{sensor.lower()}")
            cls = getattr(ClassType, sensor.upper())
        except (ModuleNotFoundError, AttributeError):
            cls = ClassType.GENERAL
        if sensor not in self.all_sensors:
            self.all_sensors[sensor] = SensorManage()
        try:
            driver_cls = ClassFactory.get_cls(cls, config['driver']['name'])
            driver = driver_cls(name=name, config=config)
        except:  # noqa
            raise SensorError(f"Initial sensor driver {name} failure, skip ...")
        self.all_sensors[sensor].add(name=name, sensor=driver)
        if len(self.all_sensors[sensor]) == 1:
            setattr(self, sensor.lower(), driver)

    def add_sensor_cls(self, sensor: str):
        try:
            _ = import_module(f"robosdk.sensors.{sensor.lower()}")
            cls = getattr(ClassType, sensor.upper())
        except (ModuleNotFoundError, AttributeError):
            cls = ClassType.GENERAL
        if sensor not in self.all_sensors:
            self.all_sensors[sensor] = SensorManage()
        for inx, cfg in enumerate(self.config.sensors[sensor]):
            _cfg = _init_cfg(config=cfg['config'], kind=sensor)
            sensor_cfg = Config(_cfg)
            sensor_cfg.update_obj(cfg)
            name = sensor_cfg["name"] or f"{sensor}{inx}"
            try:
                driver_cls = ClassFactory.get_cls(
                    cls, sensor_cfg['driver']['name'])
                driver = driver_cls(name=name, config=sensor_cfg)
            except Exception as err:  # noqa
                self.logger.error(
                    f"Initial sensor driver {name} failure : {err}, skip ...")
                return
            if inx == 0:
                setattr(self, sensor.lower(), driver)
            self.all_sensors[sensor].add(name=name, sensor=driver)
        if len(self.all_sensors[sensor]) > 1:
            self.logger.warning(
                f"Multiple {sensor}s defined in Robot {self.robot_name}.\n"
                f"In this case, {self.all_sensors[sensor].default_sensor} "
                f"is set as default. Switch the sensors excepted to use by "
                f"calling the `switch_sensor` method.")
        self.logger.info(f"Sensor {sensor} added")

    async def initial_sensors(self):
        for sensor in self.config.sensors:
            self.add_sensor_cls(sensor)

    def switch_sensor(self, sensor: str, name: str):
        driver = self.all_sensors[sensor][name]
        if driver is None:
            self.logger.error(f"Switch {sensor} fails because the "
                              f"device {name} cannot be located.")
            return False
        setattr(self, sensor.lower(), driver)
        self.all_sensors[sensor].default_sensor = name
        self.logger.info(f"Switch {sensor} to {name} as default.")
        return True

    async def initial_control(self):
        if "control" not in self.config:
            return
        for ctl_dict in self.config['control']:
            ctl = list(ctl_dict.keys())[0]
            cfg = ctl_dict[ctl]
            _cfg = _init_cfg(config=cfg['config'], kind="control")
            control = Config(_cfg)
            try:
                _ = import_module(f"robosdk.sensors.control.{ctl.lower()}")
                driver_cls = ClassFactory.get_cls(ClassType.CONTROL,
                                                  control['driver']['name'])
                driver = driver_cls(name=ctl, config=control)
            except Exception as e:
                self.logger.error(f"Initial control driver {ctl} failure, {e}")
                continue
            setattr(self, "control", driver)

    async def initial_navigation(self):
        if "navigation" not in self.config:
            return
        cfg = _init_cfg(config=self.config["navigation"]['config'],
                        kind="common")
        nav_cfg = Config(cfg)
        nav_cfg.update_obj(self.config["navigation"])
        name = nav_cfg["name"]
        try:
            _ = import_module("robosdk.algorithms.navigation")
            driver_cls = ClassFactory.get_cls(ClassType.NAVIGATION,
                                              nav_cfg['driver']['name'])
            driver = driver_cls(name=name, config=nav_cfg)
        except Exception as e:
            self.logger.error(f"Initial navigation driver {name} failure, {e}")
            return
        setattr(self, "navigation", driver)

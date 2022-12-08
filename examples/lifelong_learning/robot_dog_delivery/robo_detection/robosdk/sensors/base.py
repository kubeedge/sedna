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
import copy
import threading

from robosdk.utils.logger import logging
from robosdk.utils.util import Config
from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory


class SensorBase(metaclass=abc.ABCMeta):
    """
    This class defines an interface for sensors, it defines
    the basic functionality required by sensors being used in
    an environment.
    """

    def __init__(self, name: str, config: Config):
        self.sensor_name = name
        self.config = config
        self.interaction_mode = self.config.get("driver", {}).get("type", "UK")
        self.logger = logging.bind(instance=self.sensor_name, sensor=True)

    def connect(self):
        pass

    def close(self):
        pass

    def reset(self):
        """Reset the sensor data"""
        raise NotImplementedError("Sensor object must reset")


@ClassFactory.register(ClassType.GENERAL, alias="ros_common_driver")
class RosCommonDriver(SensorBase): # noqa

    def __init__(self, name, config: Config = None):
        super(RosCommonDriver, self).__init__(name=name, config=config)

        import rospy
        import rostopic
        import roslib.message
        import message_filters

        self.data_lock = threading.RLock()
        self.sensor_kind = name
        self.data = None
        try:
            msg_type, _, _ = rostopic.get_topic_type(self.config.data.target)
            self.message_class = roslib.message.get_message_class(msg_type)
        except rostopic.ROSTopicIOException:
            self.logger.error(f"message type - {name} were unable to load.")
            self.message_class = rospy.msg.AnyMsg
        self.sub = message_filters.Subscriber(
            self.config.data.target, self.message_class)
        self.sub.registerCallback(self.data_callback)

    def data_callback(self, data):
        self.data = data

    def get_data(self):
        return copy.deepcopy(self.data)

    def close(self):
        self.sub.sub.unregister()


class SensorManage:

    def __init__(self):
        self.default_sensor = ""
        self._all_sensors = {}

    def add(self, name: str, sensor: SensorBase = None):
        if not len(self._all_sensors):
            self.default_sensor = name
        self._all_sensors[name] = sensor

    def remove(self, name: str):
        if name in self._all_sensors:
            del self._all_sensors[name]
        if self.default_sensor == name:
            self.default_sensor = list(self._all_sensors.keys())[0] if len(self) else ""

    def __len__(self) -> int:
        return len(self._all_sensors)

    def __getitem__(self, item: str) -> SensorBase:
        return self._all_sensors.get(item, None)

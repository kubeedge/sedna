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
from typing import Any
from copy import deepcopy

from robosdk.utils.exceptions import SensorError

from .base import MessageSenderBase


class RosMessagePublish(MessageSenderBase):  # noqa
    def __init__(self):
        super(RosMessagePublish, self).__init__()

    def register(self,
                 name: str,
                 topic="/image_result",
                 msg_type="sensor_msgs/Image",
                 queue_size=5,
                 converter=None,
                 convert_param=None):
        import rospy
        import roslib.message

        if name in self.msg_mapper:
            self.logger.warning(f"{name} has been registered, try to replace")
        message_class = roslib.message.get_message_class(msg_type)
        if not message_class:
            raise SensorError(f"fail to regis {name} with "
                              f"message type {msg_type}")
        self.msg_mapper[name] = {
            "sender": rospy.Publisher(topic, message_class,
                                      queue_size=queue_size),
            "count": 0,
            "curr": None,
            "converter": converter,
            "convert_param": convert_param if isinstance(
                convert_param, dict) else {}
        }

    def send(self, name: str, data: Any):
        if data is None:
            return
        if name not in self.msg_mapper:
            self.logger.error(f"{name} has not been registered")
            return
        self.msg_mapper[name]["count"] += 1
        self.msg_mapper[name]["curr"] = deepcopy(data)
        if callable(self.msg_mapper[name]["converter"]):
            data = self.msg_mapper[name]["converter"](
                data, **self.msg_mapper[name]["convert_param"])
        self.msg_mapper[name]["sender"].publish(data)
        self.logger.info(f"frame {self.msg_mapper[name]['count']} "
                         f"of {name} send complete")

    def unregister(self, name):
        if name in self.msg_mapper:
            del self.msg_mapper[name]

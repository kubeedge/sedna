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

from robosdk.utils.util import Config
from robosdk.utils.logger import logging
from robosdk.cloud_robotics.message_channel import WSMessageChannel
from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory


class DataCollectorBase(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        self.data_sub_lock = threading.RLock()
        self.curr_frame_id = 0
        self.curr_frame = None

    @abc.abstractmethod
    def start(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def close(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def parse(self, *args, **kwargs):
        ...


class DataCollector(metaclass=abc.ABCMeta):  # noqa
    def __init__(self, config: Config = None):
        self.config = config
        self.logger = logging.bind(instance="dataCollector", system=True)
        self.data_sub_lock = threading.RLock()
        try:
            self.message_channel = ClassFactory.get_cls(
                ClassType.GENERAL, self.config.Interaction)()
        except (ValueError, AttributeError) as e:
            self.logger.warning(f"fail to locate message channel, {e}, "
                                f"use `WSMessageChannel` as default")
            self.message_channel = WSMessageChannel()

    @abc.abstractmethod
    def connect(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def close(self, *args, **kwargs):
        ...

    def start(self, *args, **kwargs):
        self.message_channel.setDaemon(True)
        self.message_channel.start()

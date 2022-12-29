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

from robosdk.utils.logger import logging


class MessageSenderBase(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        self.msg_mapper = {}
        self.logger = logging.bind(instance="message_sender")

    @abc.abstractmethod
    def register(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def unregister(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def send(self, *args, **kwargs):
        ...

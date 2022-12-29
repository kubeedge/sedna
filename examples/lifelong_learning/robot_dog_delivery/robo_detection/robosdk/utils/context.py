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
"""
  Context/System configure manager, Environment variables obtain
"""

import os
import ssl
from typing import AnyStr
from robosdk.utils.util import singleton
from robosdk.utils.util import get_machine_type


__all__ = ("Context", "BaseConfig", "MessageConfig")

_sdk_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)


class Context:
    """The Context provides the capability of obtaining the context"""
    parameters = os.environ

    @classmethod
    def get(cls, param: AnyStr, default: AnyStr = None) -> AnyStr:
        """get the value of the key `param` in `PARAMETERS`,
        if not exist, the default value is returned"""
        value = cls.parameters.get(
            param) or cls.parameters.get(str(param).upper())
        return value or default


@singleton
class BaseConfig:
    """ Base configure manager"""
    ROS_MASTER_URI = Context.get('ROS_MASTER_URI', "http://localhost:11311")
    ROBOT_ID = Context.get('ROBOT_ID', "")
    taskId = Context.get('TASK_ID', "")

    logDir = Context.get("LOG_DIR", "/tmp")
    logUri = Context.get("LOG_URI", "")
    logLevel = Context.get("LOG_LEVEL", "INFO")

    configPath = Context.get("CFG_PATH", os.path.join(_sdk_path, "configs"))

    machineType = get_machine_type()


@singleton
class MessageConfig:
    CA_CERTS = Context.get('CA_CERTS', "")
    CERT_FILE = Context.get('CERT_FILE', "")
    KEY_FILE = Context.get('KEY_FILE', "")
    CHECK_HOSTNAME = Context.get('CA_CERTS', "False").upper() == "TRUE"
    SSL_VERSION = int(
        getattr(ssl,
                Context.get('SSL_VERSION', "PROTOCOL_TLSv1_2"),
                ssl.PROTOCOL_TLSv1_2))
    CERT_REQS = int(
        getattr(ssl,
                Context.get('CERT_REQS', "CERT_NONE"),
                ssl.CERT_NONE))

    ENDPOINT = Context.get('ENDPOINT', "wss://localhost:8443/ws")
    QUEUE_MAXSIZE = int(Context.get('QUEUE_MAXSIZE', "1000"))

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

"""Base logger"""

import logging

import colorlog

from sedna.common.config import BaseConfig

LOG_LEVEL = BaseConfig.log_level


class Logger:
    """
    Deafult logger in sedna
    Args:
        name(str) : Logger name, default is 'sedna'
    """

    def __init__(self, name: str = BaseConfig.job_name):
        self.logger = logging.getLogger(name)

        self.format = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)-15s] %(filename)s(%(lineno)d)'
            ' [%(levelname)s]%(reset)s - %(message)s', )

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logLevel = 'INFO'
        self.logger.setLevel(level=LOG_LEVEL)
        self.logger.propagate = False


LOGGER = Logger().logger

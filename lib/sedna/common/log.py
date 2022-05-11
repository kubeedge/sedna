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
import json
from sedna.common.config import BaseConfig

LOG_LEVEL = BaseConfig.log_level

class JsonFormatter:
    """
    JSON logs formatter, required for application-level tracing
    """

    def format(self, record):
        formatted_record = dict()
        try:
            for key in ['created', 'levelname', 'pathname', 'funcName', 'msg']:
                formatted_record[key] = getattr(record, key)

            return json.dumps(formatted_record, indent=None)
        except TypeError:
            print(record)
            return ""

class Logger:
    """
    Deafult logger in sedna
    Args:
        name(str) : Logger name, default is 'sedna'
    """

    def __init__(self, name: str = BaseConfig.job_name):
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level=LOG_LEVEL)

        self.format = JsonFormatter()
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)
        self.handler.setLevel(level=LOG_LEVEL)

        self.logger.addHandler(self.handler)

        self.logger.propagate = False

LOGGER = Logger().logger

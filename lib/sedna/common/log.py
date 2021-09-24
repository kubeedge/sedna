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
import colorlog

from sedna.common.config import BaseConfig

# MEASUREMENT = 9
LOG_DIR = "logs/"
LOG_LEVEL = BaseConfig.log_level

# logging.addLevelName(MEASUREMENT, "MEASUREMENT")

class LogFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level

class JsonFormatter:
   def format(self, record):
       formatted_record = dict()

       for key in ['created', 'levelname', 'pathname', 'funcName', 'msg']:
           formatted_record[key] = getattr(record, key)

       return json.dumps(formatted_record, indent=None)

class Logger:
    """
    Deafult logger in sedna
    Args:
        name(str) : Logger name, default is 'sedna'
    """

    def __init__(self, name: str = BaseConfig.job_name):
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level=LOG_LEVEL)

        # self.format = colorlog.ColoredFormatter(
        #     '%(log_color)s[%(asctime)-15s] %(filename)s(%(lineno)d)'
        #     ' [%(levelname)s]%(reset)s - %(message)s', )

        self.format= JsonFormatter()
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)
        self.handler.setLevel(level=LOG_LEVEL)

        # self.logger.measurement = self.measurement

        # Create file handler
        # self.fh = logging.FileHandler('spam.log')
        # self.fh.setFormatter(self.format)
        # self.fh.addFilter(LogFilter(logging.DEBUG))
        # self.fh.setLevel(logging.DEBUG)

        self.logger.addHandler(self.handler)
        # self.logger.addHandler(self.fh)

        self.logger.propagate = False

    # def measurement(self, message, *args, **kws):
    #     if self.logger.isEnabledFor(MEASUREMENT):
    #         # Yes, logger takes its '*args' as 'args'.
            # self.logger._log(MEASUREMENT, message, args, **kws) 


LOGGER = Logger().logger

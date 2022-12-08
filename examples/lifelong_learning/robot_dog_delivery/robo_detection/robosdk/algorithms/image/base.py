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
from typing import List

from robosdk.utils.logger import logging
from robosdk.utils.util import Config
from robosdk.utils.util import ImageQualityEval


__all__ = ("Resolution", )


class Resolution(metaclass=abc.ABCMeta):
    def __init__(self, name: str, config: Config):
        self.resolution_name = name
        self.config = config
        if (getattr(self.config, "image", "") and
                hasattr(self.config.image, "eval")):
            eval_func = self.config.image.eval
        else:
            eval_func = "vollath"
        self.eval = getattr(
            ImageQualityEval, eval_func, ImageQualityEval.vollath)
        self.logger = logging.bind(instance=self.resolution_name, system=True)

    @abc.abstractmethod
    def inference(self, imgs: List):
        ...

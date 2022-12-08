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

from typing import List

import cv2
import numpy as np

from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory
from robosdk.utils.util import Config
from robosdk.utils.exceptions import SensorError

from .base import Resolution


__all__ = ("SimpleSelection", )


@ClassFactory.register(ClassType.IMAGE, alias="base_select")
class SimpleSelection(Resolution):  # noqa

    def __init__(self, name: str = "base_select", config: Config = None):
        super(SimpleSelection, self).__init__(name=name, config=config)

    def inference(self, imgs: List[np.ndarray]) -> np.ndarray:
        if not len(imgs):
            raise SensorError("No input images found")
        if len(imgs) == 1:
            return imgs[0]
        s = sorted(imgs, key=lambda j: self.eval(
            cv2.cvtColor(j, cv2.COLOR_BGR2GRAY)))
        return s[-1]

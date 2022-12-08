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
import os
import time
import threading
from importlib import import_module
from typing import Tuple, Optional, Any

import cv2
import numpy as np

from robosdk.utils.util import Config
from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory
from robosdk.sensors.base import SensorBase


__all__ = ("Camera", )


class Camera(SensorBase):  # noqa
    """
    This is a parent class on which the robot
    specific Camera classes would be built.
    """

    def __init__(self, name, config: Config = None):
        super(Camera, self).__init__(name=name, config=config)
        self.camera_info_lock = threading.RLock()
        self.camera_img_lock = threading.RLock()
        self.sensor_kind = "camera"
        self.rgb_img = None
        self.depth_img = None
        self.camera_info = None
        self.camera_P = None

    def get_rgb(self) -> Tuple[np.array, Any]:
        """
        This function returns the RGB image perceived by the camera.
        """
        raise NotImplementedError

    def get_depth(self) -> Tuple[np.array, Any]:
        """
        This function returns the depth image perceived by the camera.

        The depth image is in meters.

        :rtype: np.ndarray or None
        """
        raise NotImplementedError

    def get_rgb_depth(self) -> Tuple[np.array, Optional[np.array]]:
        """
        This function returns both the RGB and depth
        images perceived by the camera.
        The depth image is in meters.
        :rtype: np.ndarray or None
        """
        raise NotImplementedError

    def get_intrinsics(self) -> Optional[np.array]:
        """
        This function returns the camera intrinsics.

        :rtype: np.ndarray
        """
        raise NotImplementedError

    def capture(self, save_path, algorithm="base_select", expose_time=10):
        """
        This function capture the image by using the define algorithms
        """
        try:
            _ = import_module("robosdk.algorithms.image")
            driver_cls = ClassFactory.get_cls(ClassType.IMAGE, algorithm)
            driver = driver_cls(config=self.config)
            imgs = []
            for _ in range(expose_time):
                imgs.append(self.rgb_img)
                time.sleep(0.1)
            img = driver.inference(imgs)
        except Exception as e:
            self.logger.error(
                f"Initial capture algorithm {algorithm} failure, {e}")
            img = self.rgb_img
        out_dir = os.path.dirname(save_path)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        _ = cv2.imwrite(save_path, img)
        return save_path

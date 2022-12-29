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

import copy

import cv2
import numpy as np

from robosdk.utils.util import Config
from robosdk.utils.logger import logging
from robosdk.utils.context import BaseConfig
from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory

from .base import Camera

__all__ = ("RosCameraDriver",)


@ClassFactory.register(ClassType.CAMERA, alias="ros_camera_driver")
class RosCameraDriver(Camera):  # noqa

    def __init__(self, name, config: Config = None):
        super(RosCameraDriver, self).__init__(name=name, config=config)

        import rospy
        import message_filters
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image

        self.cv_bridge = CvBridge()
        self.get_time = rospy.get_rostime
        rospy.Subscriber(
            self.config.info.target,
            CameraInfo,
            self._camera_info_callback,
        )

        rgb_topic = self.config.rgb.target
        self.rgb_sub = message_filters.Subscriber(rgb_topic, Image)

        depth_topic = self.config.depth.target
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)

        img_subs = [self.rgb_sub, self.depth_sub]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            img_subs, queue_size=10, slop=0.2
        )
        self.sync.registerCallback(self._sync_callback)

    def _sync_callback(self, rgb, depth):
        try:
            if rgb is not None:
                self.rgb_img = self.cv_bridge.imgmsg_to_cv2(
                    rgb, self.config.rgb.encoding)
                if (self.config.rgb.encoding == "bgr8" and
                        BaseConfig.machineType.startswith("aarch")):
                    self.rgb_img = self.rgb_img[:, :, ::-1]
            self.depth_img = self.cv_bridge.imgmsg_to_cv2(
                depth, self.config.depth.encoding)
            self.depth_img = np.nan_to_num(self.depth_img)
        except Exception as e:
            logging.error(f"get frame data from camera fail: {str(e)}")

    def _camera_info_callback(self, msg):
        self.camera_info_lock.acquire()
        self.camera_info = msg
        self.camera_P = np.array(msg.P).reshape((3, 4))
        self.camera_info_lock.release()

    def get_rgb(self):
        """
        This function returns the RGB image perceived by the camera.
        """
        self.camera_img_lock.acquire()
        ts = self.get_time()
        rgb = copy.deepcopy(self.rgb_img)
        self.camera_img_lock.release()
        return rgb, ts

    def get_depth(self):
        """
        This function returns the depth image perceived by the camera.

        The depth image is in meters.

        :rtype: np.ndarray or None
        """
        self.camera_img_lock.acquire()
        ts = self.get_time()
        depth = copy.deepcopy(self.depth_img)
        self.camera_img_lock.release()
        if self.depth_img is None:
            return None, ts
        if self.config.depth.map_factor:
            depth = depth / self.config.depth.map_factor
        else:
            depth = cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX)
        return depth, ts

    def get_rgb_depth(self):
        """
        This function returns both the RGB and depth
        images perceived by the camera.
        The depth image is in meters.
        :rtype: np.ndarray or None
        """
        self.camera_img_lock.acquire()
        rgb = copy.deepcopy(self.rgb_img)
        depth = copy.deepcopy(self.depth_img)
        if depth is not None:
            if self.config.depth.map_factor:
                depth = depth / self.config.depth.map_factor
            else:
                depth = cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX)
        self.camera_img_lock.release()
        return rgb, depth

    def get_intrinsics(self):
        """
        This function returns the camera intrinsics.

        :rtype: np.ndarray
        """
        if self.camera_P is None:
            return self.camera_P
        self.camera_info_lock.acquire()
        p = copy.deepcopy(self.camera_P)
        self.camera_info_lock.release()
        return p[:3, :3]

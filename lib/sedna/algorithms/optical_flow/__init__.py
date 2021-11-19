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

"""Optical Flow Algorithms"""
import abc

import numpy
import cv2
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.benchmark import FTimer
from sedna.common.log import LOGGER

__all__ = ('LukasKanade')


class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define unified interface."""

    def __call__(self, old_frame=None, current_frame=None):
        """predict function, and it must be implemented by
        different methods class.

        :param old_frame: prev_image to compare against
        :param current_frame: next_image to check for motion
        :return: `True` means that there is movement in two subsequent frames, `False` means that there is no movement.
        """
        raise NotImplementedError

@ClassFactory.register(ClassType.OF, alias="LukasKanadeOF")
class LukasKanade(BaseFilter, abc.ABC):
    def __init__(self, **kwargs):
        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Parameters for Lucas Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def __call__(self, old_frame=None, current_frame=None):
        """
        :param old_img: prev_image to compare against
        :param current_img: next_image to check for motion
        :return: `True` means that there is movement in two subsequent frames, `False` means that there is no movement.
        """
        with FTimer(f"LukasKanadeOF"):
            movement = False
            try:
                old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                # Calculate Optical Flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, current_gray, p0, None, **self.lk_params
                )

                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # We perform rounding because there might ba a minimal difference 
                # even between two images of the same subject (image compared against itself)
                # Allclose is used instead of array_equal to support array of floats (if we remove rounding).
                movement = not numpy.allclose(numpy.rint(good_new), numpy.rint(good_old))
            except Exception as ex:
                LOGGER.error(f"Error during the execution of the optical flow estimation! [{ex}]")

            return movement
        
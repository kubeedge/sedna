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
import cv2

from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('LukasKanade')


class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define unified interface."""

    def __call__(self, old_img=None, current_img=None):
        """predict function, and it must be implemented by
        different methods class.

        :param old_img: prev_image to compare against
        :param current_img: next_image to check for motion
        :return: `True` means that there is movement in two subsequent frames, `False` means that there is no movement.
        """
        raise NotImplementedError

    @classmethod
    def data_check(cls, data):
        """Check the data in [0,1]."""
        return 0 <= float(data) <= 1


@ClassFactory.register(ClassType.OF, alias="LukasKanade")
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

        # Create random colors
        # self.color = np.random.randint(0, 255, (100, 3))



    def __call__(self, old_frame=None, current_frame=None):
        """
        :param old_img: prev_image to compare against
        :param current_img: next_image to check for motion
        :return: `True` means that there is movement in two subsequent frames, `False` means that there is no movement.
        """
        # if invalid input, return False
        if not old_frame or not current_frame:
            return False

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

        # # Draw the tracks
        # for i, (new, old) in enumerate(zip(good_new, good_old)):
        #     a, b = new.ravel()
        #     c, d = old.ravel()
        #     mask = cv2.line(mask, (a, b), (c, d), self.color[i].tolist(), 2)
        #     frame = cv2.circle(frame, (a, b), 5, self.color[i].tolist(), -1)

        # Does this work?
        if good_new == good_old or err:
            return False
        else:
            return True
        
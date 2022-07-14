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
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.log import LOGGER

__all__ = ('LukasKanade')


class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define an unified interface."""

    def __call__(self, old_frame=None, current_frame=None):
        """predict function, and it must be implemented by
        different methods class.

        :param old_frame: prev_image to compare against
        :param current_frame: next_image to check for motion
        :return: `True` means that there is movement in two subsequent
            frames, `False` means that there is no movement.
        """
        raise NotImplementedError


@ClassFactory.register(ClassType.OF, alias="LukasKanadeOF")
class LukasKanade(BaseFilter, abc.ABC):
    import cv2

    """
        Class to detect movement between two consecutive images.
    """
    def __init__(self, **kwargs):
        # Parameters for ShiTomasi corner detection
        self.feature_params = \
            dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Parameters for Lucas Kanade optical flow
        _criteria = self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(_criteria, 10, 0.03),
        )

    def __call__(self, old_frame=None, current_frame=None):
        """
        :param old_img: prev_image to compare against
        :param current_img: next_image to check for motion
        :return: `True` means that there is movement in two subsequent frames,
             `False` means that there is no movement.
        """

        movement = False
        try:
            old_gray = self.cv2.cvtColor(old_frame, self.cv2.COLOR_BGR2GRAY)
            p0 = self.cv2.goodFeaturesToTrack(
                old_gray, mask=None, **self.feature_params)

            current_gray = \
                self.cv2.cvtColor(current_frame, self.cv2.COLOR_BGR2GRAY)

            # Calculate Optical Flow
            p1, st, err = self.cv2.calcOpticalFlowPyrLK(
                old_gray, current_gray, p0, None, **self.lk_params
            )

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # We perform rounding because there might ba a minimal difference
            # even between two images of the same subject
            # (image compared against itself)
            # Allclose is used instead of array_equal to support
            # array of floats (if we remove rounding).
            movement = \
                not numpy.allclose(
                    numpy.rint(good_new),
                    numpy.rint(good_old)
                )
        except Exception as ex:
            LOGGER.error(
                f"Error during the execution of\
                     the optical flow estimation! [{ex}]")

        return movement


@ClassFactory.register(ClassType.OF, alias="LukasKanadeOF_CUDA")
class LukasKanadeCUDA(BaseFilter, abc.ABC):
    import cv2
    import numpy

    """
        Class to detect movement between
        two consecutive images (GPU implementation).
    """
    def __init__(self, **kwargs):
        # Parameters for ShiTomasi corner detection
        self.feature_params = \
            dict(
                srcType=self.cv2.CV_8UC1,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7)

        # Parameters for Lucas Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2
        )

        self.corner_detector = \
            self.cv2.cuda.createGoodFeaturesToTrackDetector(
                **self.feature_params
            )
        self.of = \
            self.cv2.cuda.SparsePyrLKOpticalFlow_create(**self.lk_params)

    def __call__(self, old_frame=None, current_frame=None):
        """
        :param old_img: prev_image to compare against
        :param current_img: next_image to check for motion
        :return: `True` means that there is movement in two subsequent frames,
             `False` means that there is no movement.
        """

        old_frame = self.cv2.cuda_GpuMat(old_frame)
        current_frame = self.cv2.cuda_GpuMat(current_frame)

        movement = False
        try:
            old_gray = \
                self.cv2.cuda.cvtColor(old_frame, self.cv2.COLOR_BGR2GRAY)
            p0 = self.corner_detector.detect(old_gray)

            current_gray = \
                self.cv2.cuda.cvtColor(current_frame, self.cv2.COLOR_BGR2GRAY)

            # Calculate Optical Flow
            p1, st, err = self.of.calc(
                old_gray, current_gray, p0, None
            )

            # Select good points
            p0 = p0.download().astype(numpy.float32)
            p1 = p1.download().astype(numpy.float32)
            st = st.download().astype(numpy.float32)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # We perform rounding because there might ba a minimal difference
            # even between two images of the same subject
            # (image compared against itself)
            # Allclose is used instead of array_equal to
            # support array of floats (if we remove rounding).
            movement = \
                not numpy.allclose(
                    numpy.rint(good_new),
                    numpy.rint(good_old)
                )
        except Exception as ex:
            LOGGER.error(
                f"Error during the execution of\
                     the optical flow estimation! [{ex}]")

        return movement

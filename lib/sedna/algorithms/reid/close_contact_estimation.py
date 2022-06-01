# Social-Distancing
# Copyright (c) 2020 IIT PAVIS


# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE
# OR DOWNLOAD THE SOFTWARE.

# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

# Modified Copyright 2021 The KubeEdge Authors.
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

from typing import List, Tuple
import cv2
import numpy as np


class ContactTracker(object):
    """
    The ContactTracker object is invoked in frames where the
    target person was identified.
    """
    def __init__(self, draw_top_view=False) -> None:
        """

        @param h_ratio: Ratio between the closest horizontal line of the scene
        to the furthest visible. It must be a float value in (0,1)
        @param v_ratio: Ratio between the height of the trapezoid wrt the
        rectangular bird’s view scene (image height).
        It must be a float value in (0,1).
        """

        # unwrap target
        self.bbox_target = None

        # perspective information
        self.h_ratio = None
        self.v_ratio = None

        # Compute homography
        self.homography_matrix = None
        self.target_map = None
        self.ellipse_bbox_target = None
        self.ellipse_target = None

        self.CTV_L = {}
        self.distance_threshold = 1.8  # meters
        self.draw_top_view = draw_top_view
        self.top_view_exists = False

    def prep_homography(self,
                        img_shape,
                        bbox_target,
                        h_ratio: float = 0.5,
                        v_ratio: float = 0.5):

        self.h_ratio = h_ratio
        self.v_ratio = v_ratio

        if self.homography_matrix is None:
            self.compute_homography(img_shape)

        # initialize ellipse for target
        bbox, ellipse = self.create_ellipse([bbox_target])
        self.ellipse_bbox_target = bbox[0]
        self.ellipse_target = ellipse[0]

        # generate ellipse mask
        self.target_map = np.ascontiguousarray(
            np.zeros(img_shape),
            dtype=np.uint8
        )
        cv2.ellipse(self.target_map,
                    (self.ellipse_target[0], self.ellipse_target[1]),
                    (self.ellipse_target[2], self.ellipse_target[3]),
                    0, 0, 360,
                    (255, 0, 0), thickness=-1)

    def compute_homography(self, img_shape: List[int]) -> None:
        """
        Calculate homography
        @param img_shape: List [h,w]
        """

        r_height = img_shape[1] * self.v_ratio
        r_width = img_shape[0] * self.h_ratio

        src = np.array([
            [0, 0],
            [0, img_shape[1]],
            [img_shape[0], img_shape[1]],
            [img_shape[0], 0]
        ])

        dst = np.array([
            [0 + r_width/2, 0 + r_height],
            [0, img_shape[1]],
            [img_shape[0], img_shape[1]],
            [img_shape[0] - r_width/2, 0 + r_height]], np.int32)

        self.homography_matrix, status = cv2.findHomography(src, dst)

    def in_risk_zone(self,
                     img: np.ndarray,
                     bbox_candidate: List[int] = None) -> bool:

        # keep default color to green - no contact
        color = (0, 255, 0)
        in_contact = False

        # check if homography matrix was properly initialized or not
        if self.homography_matrix is None:
            self.compute_homography([img.shape[0], img.shape[1]])

        t_ellipse_bbox_candidate, t_ellipse_candidate = \
            self.create_ellipse([bbox_candidate])
        ellipse_bbox_candidate = t_ellipse_bbox_candidate[0]
        ellipse_candidate = t_ellipse_candidate[0]

        # check for overlap between bounding boxes: coarse check
        has_overlap = self.check_bbox_overlap(
            ellipse_bbox_candidate, self.ellipse_bbox_target)

        if has_overlap:
            # check if the contours intersect or not: finer check
            temp = np.ascontiguousarray(
                np.zeros(img.shape), dtype=np.uint8
            )

            # creating mask for candidate
            cv2.ellipse(temp,
                        (ellipse_candidate[0], ellipse_candidate[1]),
                        (ellipse_candidate[2], ellipse_candidate[3]),
                        0, 0, 360,
                        (255, 0, 0), thickness=-1)

            olap = np.sum(
                np.multiply(self.target_map, temp).reshape(-1)
            )

            # if contours don't intersect, they are not in contact
            if not olap == 0:
                in_contact = True

                # change color to red
                color = (255, 0, 0)

        # thickness set to negative fills the eclipse
        # draw image inplace - once for candidate, once for target
        # for target - ellipse color is always red

        cv2.ellipse(img,
                    (self.ellipse_target[0], self.ellipse_target[1]),
                    (self.ellipse_target[2], self.ellipse_target[3]),
                    0, 0, 360,
                    (255, 0, 0), thickness=2)

        cv2.ellipse(img,
                    (ellipse_candidate[0], ellipse_candidate[1]),
                    (ellipse_candidate[2], ellipse_candidate[3]),
                    0, 0, 360,
                    color, thickness=2)

        return in_contact

    def create_ellipse(
            self, bbox_list: List[List[int]] = None) -> Tuple[List, List]:
        """
        Create ellipses for each of the generated bounding boxes.
        @param bbox_list:
        """

        # initialize placeholder
        ellipse_bboxes = []
        draw_ellipse_requirements = []

        for i, box in enumerate(bbox_list):
            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            left, right, top, bottom = x0, x1, y0, y1

            # evaluate the height using the bounding box directly
            height = x1 - x0

            # calculate bounding box center
            bbox_center = np.array(
                [(x0 + x1) // 2, (y0 + y1)//2], np.int
            )

            # computing how the height of the circle varies in perspective
            pts = np.array([
                [bbox_center[0], top],
                [bbox_center[0], bottom]], np.float32)
            pts1 = pts.reshape(-1, 1, 2).astype(np.float32)  # n,1,2

            dst1 = cv2.perspectiveTransform(pts1, self.homography_matrix)

            width = int(dst1[1, 0][1] - dst1[0, 0][1])

            if width > 0.5 * height:
                width = int(0.5 * height)

            # bounding box surrounding the ellipse,
            ellipse_bbox = (
                bbox_center[0] - height,
                bbox_center[0] + height,
                bottom - width,
                bottom + width
            )

            ellipse_bboxes.append(ellipse_bbox)

            ellipse = [int(bbox_center[0]), bottom, height, width]

            draw_ellipse_requirements.append(ellipse)

        return ellipse_bboxes, draw_ellipse_requirements

    def check_bbox_overlap(self, ellipse1: Tuple, ellipse2: Tuple) -> bool:
        """
        Check if ellipse bounding rectangles overlap or not.
        Args:
            ellipse1 (tuple): ellipse one
            ellipse2 (tuple): ellipse two
        Returns:
            boolean:
        """

        r1 = self.to_rectangle(ellipse1)
        r2 = self.to_rectangle(ellipse2)

        if (r1[0] >= r2[2]) or (r1[2] <= r2[0]) \
                or (r1[3] <= r2[1]) or (r1[1] >= r2[3]):
            return False
        else:
            return True

    def to_rectangle(self, ellipse: Tuple) -> Tuple:
        """Convert ellipse to rectangle (top, left, bottom, right)
        Args:
            ellipse (tuple): bounding rectangle descriptor
        """
        x, y, a, b = ellipse

        return (x - a,
                y - b,
                x + a,
                y + b)

    def get_homography_matrix(self):
        return self.homography_matrix

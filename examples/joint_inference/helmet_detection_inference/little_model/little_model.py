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

import time
import copy
import logging

import cv2
import numpy as np

from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.core.joint_inference import JointInference

from interface import Estimator


LOG = logging.getLogger(__name__)

camera_address = Context.get_parameters('video_url')

class_names = ['person', 'helmet', 'helmet_on', 'helmet_off']
all_output_path = Context.get_parameters(
    'all_examples_inference_output'
)
hard_example_edge_output_path = Context.get_parameters(
    'hard_example_edge_inference_output'
)
hard_example_cloud_output_path = Context.get_parameters(
    'hard_example_cloud_inference_output'
)

FileOps.clean_folder([
    all_output_path,
    hard_example_cloud_output_path,
    hard_example_edge_output_path
], clean=False)


def draw_boxes(img, bboxes, colors, text_thickness, box_thickness):
    img_copy = copy.deepcopy(img)

    line_type = 2
    #  get color code
    colors = colors.split(",")
    colors_code = []
    for color in colors:
        if color == 'green':
            colors_code.append((0, 255, 0))
        elif color == 'blue':
            colors_code.append((255, 0, 0))
        elif color == 'yellow':
            colors_code.append((0, 255, 255))
        else:
            colors_code.append((0, 0, 255))

    label_dict = {i: label for i, label in enumerate(class_names)}

    for bbox in bboxes:
        if float("inf") in bbox or float("-inf") in bbox:
            continue
        label = int(bbox[5])
        score = "%.2f" % round(bbox[4], 2)
        text = label_dict.get(label) + ":" + score
        p1 = (int(bbox[1]), int(bbox[0]))
        p2 = (int(bbox[3]), int(bbox[2]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue
        try:
            cv2.rectangle(img_copy, p1[::-1], p2[::-1], colors_code[label],
                          box_thickness)
            cv2.putText(img_copy, text, (p1[1], p1[0] + 20 * (label + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                        text_thickness, line_type)
        except TypeError as err:
            # error message from pyopencv,  cv2.circle only can accept centre
            # coordinates precision up to float32. If the coordinates are in
            # float64, it will throw this error.
            LOG.warning(f"Draw box fail: {err}")
    return img_copy


def output_deal(
        final_result,
        is_hard_example,
        cloud_result,
        edge_result,
        nframe,
        img_rgb
):
    # save and show image
    img_rgb = np.array(img_rgb)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    collaboration_frame = draw_boxes(img_rgb, final_result,
                                     colors="green,blue,yellow,red",
                                     text_thickness=None,
                                     box_thickness=None)

    cv2.imwrite(f"{all_output_path}/{nframe}.jpeg", collaboration_frame)

    # save hard example image to dir
    if not is_hard_example:
        return

    if cloud_result is not None:
        cv2.imwrite(f"{hard_example_cloud_output_path}/{nframe}.jpeg",
                    collaboration_frame)
    edge_collaboration_frame = draw_boxes(
        img_rgb,
        edge_result,
        colors="green,blue,yellow,red",
        text_thickness=None,
        box_thickness=None)
    cv2.imwrite(f"{hard_example_edge_output_path}/{nframe}.jpeg",
                edge_collaboration_frame)


def main():

    # get hard exmaple mining algorithm from config
    hard_example_mining = JointInference.get_hem_algorithm_from_config(
        threshold_img=0.9
    )

    inference_instance = JointInference(
        estimator=Estimator,
        hard_example_mining=hard_example_mining
    )

    camera = cv2.VideoCapture(camera_address)
    fps = 10
    nframe = 0
    while 1:
        ret, input_yuv = camera.read()
        if not ret:
            LOG.info(
                f"camera is not open, camera_address={camera_address},"
                f" sleep 5 second.")
            time.sleep(5)
            camera = cv2.VideoCapture(camera_address)
            continue

        if nframe % fps:
            nframe += 1
            continue

        img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_BGR2RGB)
        nframe += 1
        LOG.info(f"camera is open, current frame index is {nframe}")
        is_hard_example, final_result, edge_result, cloud_result = (
            inference_instance.inference(img_rgb)
        )
        output_deal(
            final_result,
            is_hard_example,
            cloud_result,
            edge_result,
            nframe,
            img_rgb
        )


if __name__ == '__main__':
    main()

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
import warnings

import cv2
import numpy as np

from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.core.incremental_learning import IncrementalLearning
from interface import Estimator


he_saved_url = Context.get_parameters("HE_SAVED_URL", '/tmp')
rsl_saved_url = Context.get_parameters("RESULT_SAVED_URL", '/tmp')
class_names = ['person', 'helmet', 'helmet_on', 'helmet_off']

FileOps.clean_folder([he_saved_url, rsl_saved_url], clean=False)


def draw_boxes(img, labels, scores, bboxes, class_names, colors):
    line_type = 2
    text_thickness = 1
    box_thickness = 1
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
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        if float("inf") in bbox or float("-inf") in bbox:
            continue
        label = int(labels[i])
        score = "%.2f" % round(scores[i], 2)
        text = label_dict.get(label) + ":" + score
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue
        try:
            cv2.rectangle(img, p1[::-1], p2[::-1],
                          colors_code[labels[i]], box_thickness)
            cv2.putText(img, text, (p1[1], p1[0] + 20 * (label + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                        text_thickness, line_type)
        except TypeError as err:
            warnings.warn(f"Draw box fail: {err}")
    return img


def output_deal(is_hard_example, infer_result, nframe, img_rgb):
    # save and show image
    img_rgb = np.array(img_rgb)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    colors = 'yellow,blue,green,red'

    lables, scores, bbox_list_pred = infer_result
    img = draw_boxes(img_rgb, lables, scores, bbox_list_pred, class_names,
                     colors)
    if is_hard_example:
        cv2.imwrite(f"{he_saved_url}/{nframe}.jpeg", img)
    cv2.imwrite(f"{rsl_saved_url}/{nframe}.jpeg", img)


def mkdir(path):
    path = path.strip()
    path = path.rstrip()
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def deal_infer_rsl(model_output):
    all_classes, all_scores, all_bboxes = model_output
    rsl = []
    for c, s, bbox in zip(all_classes, all_scores, all_bboxes):
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1], bbox[0], bbox[3], bbox[2]
        rsl.append(bbox.tolist() + [s, c])
    return rsl


def run():
    camera_address = Context.get_parameters('video_url')

    # get hard exmaple mining algorithm from config
    hard_example_mining = IncrementalLearning.get_hem_algorithm_from_config(
        threshold_img=0.9
    )

    input_shape_str = Context.get_parameters("input_shape")
    input_shape = tuple(int(v) for v in input_shape_str.split(","))
    # create Incremental Learning instance
    incremental_instance = IncrementalLearning(
        estimator=Estimator, hard_example_mining=hard_example_mining
    )
    # use video streams for testing
    camera = cv2.VideoCapture(camera_address)
    fps = 10
    nframe = 0
    # the input of video stream
    while 1:
        ret, input_yuv = camera.read()
        if not ret:
            time.sleep(5)
            camera = cv2.VideoCapture(camera_address)
            continue

        if nframe % fps:
            nframe += 1
            continue

        img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_BGR2RGB)
        nframe += 1
        if nframe % 1000 == 1:  # logs every 1000 frames
            warnings.warn(f"camera is open, current frame index is {nframe}")
        results, _, is_hard_example = incremental_instance.inference(
            img_rgb, post_process=deal_infer_rsl, input_shape=input_shape)
        output_deal(is_hard_example, results, nframe, img_rgb)


if __name__ == "__main__":
    run()

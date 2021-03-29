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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import time

import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image

from yolo3_multiscale import YOLOInference

LOG = logging.getLogger(__name__)


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def init_yolo(model_path, input_shape):
    print('model_path : ', model_path)

    # initialize the session and bind the corresponding graph
    yolo_graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    yolo_session = tf.Session(graph=yolo_graph, config=config)

    # initialize yoloInference object
    yolo_infer = YOLOInference(yolo_session, model_path, input_shape)

    return yolo_infer, yolo_session


def validate(model_path, test_dataset, class_names, input_shape=(352, 640)):
    yolo_infer, yolo_session = init_yolo(model_path, input_shape)

    folder_out = 'result'
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    count_img = 0
    time_all = 0.0

    class_num = len(class_names)
    count_correct = [1e-6 for ix in range(class_num)]
    count_ground = [1e-6 for ix in range(class_num)]
    count_pred = [1e-6 for ix in range(class_num)]

    for line in test_dataset:
        line = line.strip()
        if not line:
            print("read line error")
            continue

        pos = line.find(' ')
        if pos == -1:
            print('error line : ', line)
            continue

        img_file = line[:pos]

        bbox_list_ground = line[pos + 1:].split(' ')
        time_predict, correct, pred, ground = validate_img_file(yolo_infer, yolo_session, img_file,
                                                                bbox_list_ground,
                                                                folder_out, class_names)

        count_correct = [count_correct[ix] + correct[ix] for ix in range(class_num)]
        count_pred = [count_pred[ix] + pred[ix] for ix in range(class_num)]
        count_ground = [count_ground[ix] + ground[ix] for ix in range(class_num)]

        count_img += 1
        time_all += time_predict

    print('count_correct', count_correct)
    print('count_pred', count_pred)
    print('count_ground', count_ground)

    precision = [float(count_correct[ix]) / float(count_pred[ix]) for ix in range(class_num)]
    recall = [float(count_correct[ix]) / float(count_ground[ix]) for ix in range(class_num)]

    all_precisions = sum(count_correct) / sum(count_pred)
    all_recalls = sum(count_correct) / sum(count_ground)

    print('precisions : ', precision)
    print('recalls : ', recall)

    print('all precisions : ', all_precisions)
    print('all recalls : ', all_recalls)

    print('average time = %.4f' % (time_all / float(count_img)))
    print('total time %.4f for %d images' % (time_all, count_img))
    return precision, recall, all_precisions, all_recalls


def validate_img_file(yolo_infer, yolo_session, img_file, bbox_list_ground, folder_out, class_names):
    print('validate_img_file : ', img_file)
    class_num = len(class_names)
    img_data = Image.open(img_file)
    height, width = img_data.size
    img_data = np.array(img_data)
    if len(img_data.shape) == 3:
        input_image = img_data
    elif len(img_data.shape) == 2:
        input_image = Image.new('RGB', (height, width))
        input_image = np.array(input_image)
        input_image[:, :, 0] = img_data
        input_image[:, :, 1] = img_data
        input_image[:, :, 2] = img_data
    else:
        raise ValueError('validate image file should have three channels')
    channels = input_image.shape[-1]
    if channels != 3:
        time_start = time.time()
        count_correct = [0 for ix in range(class_num)]
        count_ground = [0 for ix in range(class_num)]
        count_pred = [0 for ix in range(class_num)]
        time_predict = time.time() - time_start
        return time_predict, count_correct, count_ground, count_pred

    time_start = time.time()
    labels, scores, bbox_list_pred = yolo_infer.predict(yolo_session, input_image)
    time_predict = time.time() - time_start
    colors = 'yellow,blue,green,red'
    if folder_out is not None:
        img = draw_boxes(img_data, labels, scores, bbox_list_pred, class_names, colors)
        img_file = img_file.split("/")[-1]
        cv2.imwrite(os.path.join(folder_out, img_file), img)

    count_correct = [0 for ix in range(class_num)]
    count_ground = [0 for ix in range(class_num)]
    count_pred = [0 for ix in range(class_num)]

    count_ground_all = len(bbox_list_ground)
    count_pred_all = bbox_list_pred.shape[0]

    for ix in range(count_ground_all):
        class_ground = int(bbox_list_ground[ix].split(',')[4])
        count_ground[class_ground] += 1

    for iy in range(count_pred_all):
        bbox_pred = [bbox_list_pred[iy][1], bbox_list_pred[iy][0], bbox_list_pred[iy][3], bbox_list_pred[iy][2]]

        LOG.debug(f'count_pred={count_pred}, labels[iy]={labels[iy]}')
        count_pred[labels[iy]] += 1
        for ix in range(count_ground_all):
            bbox_ground = [int(x) for x in bbox_list_ground[ix].split(',')]
            class_ground = bbox_ground[4]

            if labels[iy] == class_ground:
                iou = calc_iou(bbox_pred, bbox_ground)
                if iou >= 0.5:
                    count_correct[class_ground] += 1
                    break

    return time_predict, count_correct, count_pred, count_ground


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def draw_boxes(img, labels, scores, bboxes, class_names, colors):
    line_type = 2
    text_thickness = 1
    box_thickness = 1
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
        cv2.rectangle(img, p1[::-1], p2[::-1], colors_code[labels[i]], box_thickness)
        cv2.putText(img, text, (p1[1], p1[0] + 20 * (label + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                    text_thickness, line_type)

    return img


def calc_iou(bbox_pred, bbox_ground):
    """user-define function for calculating the IOU of two matrixes. The
        input parameters are rectangle diagonals
    """
    x1 = bbox_pred[0]
    y1 = bbox_pred[1]
    width1 = bbox_pred[2] - bbox_pred[0]
    height1 = bbox_pred[3] - bbox_pred[1]

    x2 = bbox_ground[0]
    y2 = bbox_ground[1]
    width2 = bbox_ground[2] - bbox_ground[0]
    height2 = bbox_ground[3] - bbox_ground[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        iou = 0
    else:
        area = width * height
        area1 = width1 * height1
        area2 = width2 * height2
        iou = area * 1. / (area1 + area2 - area)

    return iou

#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

# from sedna.common.config import Context
# from sedna.common.file_ops import FileOps
# from sedna.core.incremental_learning import IncrementalLearning
# from interface import Estimator

# he_saved_url = Context.get_parameters("HE_SAVED_URL", '/tmp')
# rsl_saved_url = Context.get_parameters("RESULT_SAVED_URL", '/tmp')

# class_names = ['person', 'helmet', 'helmet_on', 'helmet_off']

# car, bus, truck
coco_car_classes = [2, 5, 7]

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
        # try:
            cv2.rectangle(img, p1[::-1], p2[::-1],
                          colors_code[labels[i]], box_thickness)
            cv2.putText(img, text, (p1[1], p1[0] + 20 * (label + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                        text_thickness, line_type)
        # except TypeError as err:
        #     warnings.warn(f"Draw box fail: {err}")
    return img

def output_deal(infer_result, nframe, img_rgb, class_names):
# def output_deal(is_hard_example, infer_result, nframe, img_rgb):
    # save and show image
    img_rgb = np.array(img_rgb)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    colors = 'yellow,blue,green,red'

    lables, scores, bbox_list_pred = infer_result
    img = draw_boxes(img_rgb, lables, scores, bbox_list_pred, class_names,
                     colors)
    # if is_hard_example:
    #     cv2.imwrite(f"{he_saved_url}/{nframe}.jpeg", img)
    # cv2.imwrite(f"{rsl_saved_url}/{nframe}.jpeg", img)


def mkdir(path):
    path = path.strip()
    path = path.rstrip()
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def post_process(model_output):
    all_classes, all_scores, all_bboxes = model_output
    rsl = []
    for c, s, bbox in zip(all_classes, all_scores, all_bboxes):
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1], bbox[0], bbox[3], bbox[2]
        rsl.append(bbox.tolist() + [s, c])
    return rsl




























# #
# # Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import os
# import cv2
# import time
# import numpy as np

# colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
#           [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
#           [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
#           [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
#           [255, 0, 170], [255, 0, 85], [85, 85, 255], [170, 170, 255], [170, 255, 170]]
# cnt_colors = len(colors)

# # car, bus, truck
# coco_car_classes = [2, 5, 7]


# def nms(boxes, scores, nms_thr):
#     """Single class NMS implemented in Numpy."""
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]

#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)

#         inds = np.where(ovr <= nms_thr)[0]
#         order = order[inds + 1]

#     return keep


# def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
#     """Multiclass NMS implemented inNumpy. Class-agnostic version."""
#     cls_inds = scores.argmax(1)
#     cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

#     valid_score_mask = cls_scores > score_thr
#     if valid_score_mask.sum() == 0:
#         return None
#     valid_scores = cls_scores[valid_score_mask]
#     valid_boxes = boxes[valid_score_mask]
#     valid_cls_inds = cls_inds[valid_score_mask]
#     keep = nms(valid_boxes, valid_scores, nms_thr)
#     if keep:
#         dets = np.concatenate(
#             [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
#         )
#     return dets


# def decode_outputs(outputs, img_size):
#     grids = []
#     expanded_strides = []

#     strides = [8, 16, 32]
#     hsizes = [img_size[0] // stride for stride in strides]
#     wsizes = [img_size[1] // stride for stride in strides]

#     for hsize, wsize, stride in zip(hsizes, wsizes, strides):
#         xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
#         grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
#         grids.append(grid)
#         shape = grid.shape[:2]
#         expanded_strides.append(np.full((*shape, 1), stride))

#     grids = np.concatenate(grids, 1)
#     expanded_strides = np.concatenate(expanded_strides, 1)
#     outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
#     outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

#     return outputs


# def postprocess(image_pred, input_shape, num_classes, conf_thre=0.3, nms_thre=0.45, ratio=1.0):
#     predictions = decode_outputs(image_pred, input_shape)

#     boxes = predictions[:, :4]
#     scores = predictions [:, 4:5] * predictions[:, 5:]

#     boxes_xyxy = np.ones_like(boxes)
#     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
#     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
#     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
#     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
#     boxes_xyxy /= ratio
#     detections = multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thre, conf_thre)

#     return detections


# def draw_bbox(image, results):
#     h, w, c = image.shape
#     for bbox in results:
#         x1, y1, x2, y2, score, label = bbox

#         label = int(label)
#         if label not in coco_car_classes:
#             continue
        
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(x2, w)
#         y2 = min(y2, h)
#         score = str(score)[:4]
#         color = (0, 0, 255)
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
#     return image




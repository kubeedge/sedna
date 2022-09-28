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

import _flowunit as modelbox
import numpy as np 
import json 
import cv2 

from yolox_utils import *

class YoloXPost(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.class_names = config.get_string_list('labels', [])

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    
    def process(self, data_context):
        in_image = data_context.input("in_image")
        in_classes = data_context.input("in_classes")
        in_scores = data_context.input("in_scores")
        in_boxes = data_context.input("in_boxes")

        out_image = data_context.output("out_image")

        for buffer_img, buffer_classes, buffer_scores, buffer_boxes in zip(in_image, in_classes, in_scores, in_boxes):
            width = buffer_img.get('width')
            height = buffer_img.get('height')
            channel = buffer_img.get('channel')
            frame_index = buffer_img.get('index')
            modelbox.debug("get frame index: {}".format(self.frame_index))

            # reshape for input_img data
            input_img = np.array(buffer_img.as_object(), copy=False)
            input_img = input_img.reshape((height, width, channel))
            img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            classes = np.array(buffer_classes.as_object(), copy=False)
            scores = np.array(buffer_scores.as_object(), copy=False)
            boxes = np.array(buffer_boxes.as_object(), copy=False)
            output = [classes, scores, boxes]
            results = post_process(deepcopy(output))
            img = output_deal(results, self.frame_index, img_rgb, class_names)
            # img = output_img(is_hard_example, results, self.frame_index, img_rgb)

            out_buffer = modelbox.Buffer(self.get_bind_device(), img)
            out_buffer.copy_meta(buffer_img)
            out_image.push_back(out_buffer)
            
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()




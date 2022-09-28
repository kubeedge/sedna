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


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import _flowunit as modelbox

class Image_shapeFlowUnit(modelbox.FlowUnit):
    # Derived from modelbox.FlowUnit
    def __init__(self):
        super().__init__()

    def open(self, config):
        # Open the flowunit to obtain configuration information
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # Process the data
        # input data
        buffer_img = data_context.input("input1")
        # output data
        out_shape = data_context.output("output1")
        
        cnt  = 1
        for buffer_shape in zip(buffer_img):
            if(cnt == 2) break;
            cnt++;
            width = buffer_shape.get('width')
            height = buffer_shape.get('height')
            w = width.as_object()
            h = height.as_object()
            shape = [h,w]
            # channel = buffer_img.get('channel')
            # frame_index = buffer_img.get('index')

            out_buffer = modelbox.Buffer(self.get_bind_device(), shape)
            out_buffer.copy_meta(buffer_img)
            out_shape.push_back(out_buffer)

        
        # Example process code.
        # Remove the following code and add your own code here.
        # for buffer in in_data:
        #     response = "Hello World " + buffer.as_object()
        #     result = response.encode('utf-8').strip()
        #     add_buffer = modelbox.Buffer(self.get_bind_device(), result)
        #     out_data.push_back(add_buffer)

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        # Close the flowunit
        return modelbox.Status()

    def data_pre(self, data_context):
        # Before streaming data starts
        return modelbox.Status()

    def data_post(self, data_context):
        # After streaming data ends
        return modelbox.Status()
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
import datetime
import pickle
import torch
import numpy as np
import cv2

from sedna.core.multi_edge_tracking.data_classes import DetTrackResult
from sedna.common.config import Context
from sedna.common.benchmark import FTimer, FluentdHelper
from sedna.common.log import LOGGER
from utils.utils import *

os.environ['BACKEND_TYPE'] = 'TORCH'
image_size = Context.get_parameters('input_shape') # in pixels!

class Yolov5(FluentdHelper):
    def __init__(self, **kwargs):
        # Initialize
        super(Yolov5, self).__init__()
        LOGGER.info("Starting object detection module")
        self.device = self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.img_size = int(image_size)
        self.camera_code = kwargs.get('camera_code', 0)
  
    def load(self, model_url="", mmodel_name=None, **kwargs):
        self.model = attempt_load(model_url, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        
    def evaluate(self, **kwargs):
        LOGGER.debug(f"Evaluating model")
        self.model.eval()

    def write_to_fluentd(self, result : DetTrackResult):
        try:
            msg = {
                    "worker": "l2-object-detector",
                    "outbound_data": len(pickle.dumps(result)),
                    "confidence": np.median(result.confidence).item()
            }

            self.send_json_msg(msg)
        except Exception as ex:
            LOGGER.error(f"Error while transmitting data to fluentd. Details: [{ex}]")


    def predict(self, data, **kwargs):
        LOGGER.debug("Manipulating source image")
        img = letterbox(data, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        LOGGER.debug("Loading image to device")
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        with torch.no_grad():
            with FTimer("detection"):
                pred = self.model(img, augment=False, visualize=False)[0]

        # NMS (this steps should output the tensor)
        with FTimer("nms"):
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        # # Process predictions
        s = ""
        bbs_list = []
        result = None
        det_time = datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")

        for _, det in enumerate(pred):  # detections per image
            imc = data.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], data.shape).round()
                                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    crop = save_one_box(xyxy, imc, file='test.jpg', BGR=True, save=False)
                    # Perform cropped image compression to reduce size
                    crop_encoded = np.array(cv2.imencode('.jpg', crop)[1])
                    
                    bbs_list.append([crop_encoded, conf.numpy(), self.camera_code, det_time])

        # TODO: Add the bbox coordinates
        if len(bbs_list) > 0:
            scene = np.array(cv2.imencode('.jpg', data)[1])
            result = DetTrackResult(
                bbox=[item[0] for item in bbs_list],
                scene=scene,
                confidence=[item[1] for item in bbs_list],
                detection_time=[item[3] for item in bbs_list],
                camera=[item[2] for item in bbs_list]
            )                   

            # Send some data to fluentd for monitoring
            self.write_to_fluentd(result)

            LOGGER.info(f"Found {len(bbs_list)} object/s in camera {self.camera_code}")
        
        return result

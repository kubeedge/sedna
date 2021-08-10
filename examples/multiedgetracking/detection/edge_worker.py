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

import torch
import numpy as np

from .utils import *

from sedna.common.config import Context
from sedna.common.benchmark import FTimer
from sedna.common.log import LOGGER

os.environ['BACKEND_TYPE'] = 'TORCH'

model_weights = Context.get_parameters('model_weights')
classifier = Context.get_parameters('model_classifier')
image_size = Context.get_parameters('input_shape') # in pixels!

class Estimator:
    def __init__(self):      

        # Initialize
        LOGGER.info("Starting object detection module")
        self.device = self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.weights = model_weights
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.img_size = int(image_size)
    
    def load(self, model_url="", mmodel_name=None, **kwargs):
        LOGGER.info("Loading model")
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        #LOGGER.info("Loading classifier")
        #self.modelc.load_state_dict(torch.load(classifier, map_location=self.device)['model']).to(self.device).eval()

    def evaluate(self, **kwargs):
        LOGGER.info(f"Evaluating model")
        self.model.eval()

    def predict(self, im0s, **kwargs):
        # Padded resize
        print(im0s.shape)
        LOGGER.info("Manipulating source image")
        img = letterbox(im0s, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        LOGGER.info("Loading image to device")
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


        # Second-stage classifier (optional)
        if self.classify:
            with FTimer("classify"):
                pred = apply_classifier(pred, self.modelc, img, im0s)

        # # Process predictions
        s = ""
        for i, det in enumerate(pred):  # detections per image
            imc = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                #for *xyxy, conf, cls in reversed(det):
                    #c = int(cls)  # integer class
                    #label = f'{self.names[c]} {conf:.2f}'
                    #plot_one_box(xyxy, im0s, label=label, color=colors(c, True))
                    #cv2.imwrite("test00.jpeg", im0s)
                    #save_one_box(xyxy, imc, file='test.jpg', BGR=True)
                
                # Write results
                bbs_list = []
                for conf in reversed(det):
                    bbs_list.append([imc, conf])

        #LOGGER.debug(bbs_list[0])
        #LOGGER.info(s)

        return (bbs_list)
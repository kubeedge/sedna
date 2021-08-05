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
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
from utils import non_max_suppression, apply_classifier, letterbox

from sedna.common.config import Context
from sedna.common.benchmark import FTimer
from sedna.common.log import LOGGER

os.environ['BACKEND_TYPE'] = 'TORCH'

model_weights = Context.get_parameters('model_weights')
model_name = Context.get_parameters('model_name')
classifier_name = Context.get_parameters('classifier_name')
image_size = Context.get_parameters('input_shape') # in pixels!
source = Context.get_parameters('video_url')

def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None, inplace=True):

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    ckpt = torch.load(weights, map_location=map_location)  # load
    model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

class Estimator:
    def __init__(self):      
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Initialize
        self.device = self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classify = True
        self.weights = model_weights
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.img_size = image_size
    
    def load(self, model_url="", mmodel_name=None, **kwargs):
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        self.modelc = load_classifier(name='resnet50', n=2)  # initialize
        self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(self.device).eval()

    def evaluate(self, **kwargs):
        LOGGER.info(f"Evaluating model")
        self.model.eval()

    def predict(self, im0s, **kwargs):
            # Padded resize
            img = letterbox(im0s, self.img_size, stride=self.stride)[0]

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32

            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            with FTimer(f"inference"):
                with torch.no_grad():
                    pred = self.model(img, augment=False, visualize=False)[0]

            # NMS (this steps should output the tensor)
            with FTimer(f"nms"):
                pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)


            # Second-stage classifier (optional)
            if self.classify:
                with FTimer(f"classify"):
                    pred = apply_classifier(pred, self.modelc, img, im0s)

            # # Process predictions
            # for i, det in enumerate(pred):  # detections per image

            #     imc = im0.copy()
            #     if len(det):
            #         # Rescale boxes from img_size to im0 size
            #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            #         # Write results
            #         bbs_list = []
            #         for _, conf, _ in reversed(det):
            #             bbs_list.append([imc, conf])
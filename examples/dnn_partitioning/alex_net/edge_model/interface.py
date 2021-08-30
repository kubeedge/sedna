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

import numpy as np

import sys
import os
import torch
import torch.nn as nn
import pickle
import urllib
import time
import json
from AlexNet import AlexNetConv4
import torchvision.transforms as transforms

import time
from PIL import Image

from sedna.common.config import Context
from sedna.common.log import LOGGER
from sedna.common.benchmark import FTimer

os.environ['BACKEND_TYPE'] = 'TORCH'

model_path = Context.get_parameters('model_path')
model_classes_path = Context.get_parameters('model_classes_path')
model_name = Context.get_parameters('model_name')
image_path = Context.get_parameters('image_path')

class Estimator:

    def __init__(self, **kwargs):
        LOGGER.info("Initializing cloud inference worker ...")
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, model_url="", mmodel_name=None, **kwargs):
        # The model should be provided by a CRD
        LOGGER.info(f"About to load model {model_name} with url {model_path}..")
        self.model = torch.load(model_path)
        self.model = self.model.to(self.device)

    def evaluate(self, **kwargs):
        LOGGER.info(f"Evaluating model")
        self.model.eval()

    def convert_to_list(self, data):
        return data.numpy().tolist()


    @staticmethod
    def preprocess(input_image):
        """Preprocess functions in edge model inference"""

        # resize image with unchanged aspect ratio using padding by opencv

        preprocess_fun = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess_fun(input_image)
        return input_tensor

    def predict(self, data, **kwargs):
        input_image = Image.open(image_path)
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model 
        if len(input_batch) == 0:
            return []
        else:
            input = input_batch.to(self.device)

            with FTimer(f"dnn_partitioning"):
                with torch.no_grad():
                    query_feat = self.model(input)
                    LOGGER.info(f"Tensor with features: {query_feat}")

            LOGGER.info(f"ITensor size {sys.getsizeof(query_feat.storage())}")
            # It returns a tensor, it should be transformed into a list before TX
            return self.convert_to_list(query_feat)


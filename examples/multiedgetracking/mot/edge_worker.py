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

import sys
import os

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from sedna.backend.torch.nets.nn import Backbone
from sedna.common.config import Context
from sedna.common.benchmark import FTimer
from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking import MultiObjectTracking
from sedna.core.multi_edge_tracking.multi_edge_tracking import MultiObjectTracking

os.environ['BACKEND_TYPE'] = 'TORCH'

model_weights = Context.get_parameters('edge_model_weights')
model_path = Context.get_parameters('model_path')
model_name = Context.get_parameters('model_name')
image_size = Context.get_parameters('input_shape')

class Estimator:

    def __init__(self, **kwargs):
        LOGGER.info(f"Initializing edge worker ...")
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = [int(image_size.split(",")[0]), int(image_size.split(",")[1])] 
        
        LOGGER.info(f"Expected image format is {self.image_size}")

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    
    def load(self, model_url="", mmodel_name=None, **kwargs):
        # The model should be provided by a CRD
        LOGGER.info(f"About to load model {model_name} with url {model_path}..")
        self.model = Backbone(num_classes=255, model_path=model_path, model_name=model_name, pretrain_choice="imagenet")

        # Here we load the model weights from the attached volume (.yaml)
        LOGGER.info(f"About to load weights for the model {model_name}..")
        self.model.load_param(model_weights)
        self.model = self.model.to(self.device)

    def quantize(self, layers = {torch.nn.Linear}, _dtype = torch.qint8):
        self.model = torch.quantization.quantize_dynamic(
            self.model,  # the original model
            layers,  # a set of layers to dynamically quantize
            dtype=_dtype)  # the target dtype for quantized weights

    def evaluate(self, **kwargs):
        LOGGER.info(f"Evaluating model")
        self.model.eval()

    def convert_to_list(self, data):
        return data.numpy().tolist()

    # def predict(self, data, **kwargs):
    #     if len(data) == 0:
    #         return []
    #     else:
    #         raw_imgsize = data.nbytes

    #         # We currently fetch the images from a video stream opened with OpenCV.
    #         # We need to convert the output from OpenCV into a format processable by the model.
    #         data = Image.fromarray(data)
    #         LOGGER.info('Finding ID {} ...'.format(data))
    #         input = torch.unsqueeze(self.transform(data), 0)
    #         input = input.to(self.device)

    #         with FTimer(f"feature_extraction"):
    #             with torch.no_grad():
    #                 query_feat = self.model(input)
    #                 LOGGER.info(f"Tensor with features: {query_feat}")

    #         LOGGER.info(f"Image size: {raw_imgsize} - Tensor size {sys.getsizeof(query_feat.storage())}")
    #         # It returns a tensor, it should be transformed into a list before TX
    #         return self.convert_to_list(query_feat)

    def predict(self, data, **kwargs):
        if len(data) == 0:
            return []
        else:
            # TEST: We get only the first element in the list of bboxes
            # We receive the image from the detection pod via REST API
            image_as_array = np.array(data[0][0]).astype(np.uint8)
            data = Image.fromarray(image_as_array)
            LOGGER.info('Finding ID {} ...'.format(data))
            input = torch.unsqueeze(self.transform(data), 0)
            input = input.to(self.device)

            with FTimer(f"feature_extraction"):
                with torch.no_grad():
                    query_feat = self.model(input)
                    LOGGER.info(f"Tensor with features: {query_feat}")

            LOGGER.info(f"Image size: {image_as_array.nbytes} - Tensor size {sys.getsizeof(query_feat.storage())}")
            # It returns a tensor, it should be transformed into a list before TX
            return self.convert_to_list(query_feat)

# Starting the ReID module
inference_instance = MultiObjectTracking(estimator=Estimator)
inference_instance.start()
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
import cv2

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from sedna.common.config import Context
from sedna.common.benchmark import FTimer, FluentdHelper
from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking import FEService
from sedna.core.multi_edge_tracking.multi_edge_tracking import FEService

os.environ['BACKEND_TYPE'] = 'TORCH'

image_size = Context.get_parameters('input_shape')

class Estimator(FluentdHelper):

    def __init__(self, **kwargs):
        super(Estimator, self).__init__()
        LOGGER.info(f"Starting feature extraction module")
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = [int(image_size.split(",")[0]), int(image_size.split(",")[1])] 
        
        LOGGER.debug(f"Expected image format is {self.image_size}")

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    
    def load(self, model_url="", mmodel_name=None, **kwargs):
        self.model = torch.load(model_url, map_location=torch.device(self.device))

    def quantize(self, layers = {torch.nn.Linear}, _dtype = torch.qint8):
        self.model = torch.quantization.quantize_dynamic(
            self.model,  # the original model
            layers,  # a set of layers to dynamically quantize
            dtype=_dtype)  # the target dtype for quantized weights

    def evaluate(self, **kwargs):
        LOGGER.debug(f"Evaluating feature extraction model")
        self.model.eval()

    def convert_to_list(self, data, camera_code, det_time):
        return [data.numpy().tolist(), camera_code, det_time]

    def write_to_fluentd(self, data):
        try:
            msg = {
                "worker": "l1-feature-extractor",
                "outbound_data": data
            }
            
            self.send_json_msg(msg)
        except Exception as ex:
            LOGGER.error(f"Error while transmitting data to fluentd. Details: [{ex}]")

    def predict(self, data, **kwargs):
        result = []

        if len(data) == 0:
            return result

        for d in data:
            # Perform image decoding and store in array
            image_as_array = cv2.imdecode(np.array(d[0][0]).astype(np.uint8), cv2.IMREAD_COLOR)
            conf_score = d[0][1]
            camera_code = d[0][2]
            det_time = d[0][3]
            
            data = Image.fromarray(image_as_array)
            LOGGER.debug(f'Performing feature extraction for received image')
            input = torch.unsqueeze(self.transform(data), 0)
            input = input.to(self.device)

            with FTimer(f"feature_extraction"):
                with torch.no_grad():
                    query_feat = self.model(input)
                    LOGGER.info(f"Extracted ReID features for {len(data)} object/s received from camera {camera_code}")
                    LOGGER.debug(f"Extracted tensor with features: {query_feat}")

            LOGGER.debug(f"Input image size: {image_as_array.nbytes}")
            LOGGER.debug(f"Output tensor size {sys.getsizeof(query_feat.storage())}")
            
            self.write_to_fluentd(sys.getsizeof(query_feat.storage()))

            # It returns a tensor, it should be transformed into a list before TX
            result.append(self.convert_to_list(query_feat, camera_code, det_time))

        return result 

# Starting the FE module
inference_instance = FEService(estimator=Estimator)
inference_instance.start()
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


import pickle
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

    def extract_features(self, data):
        total_data = 0

        for d in data:
            det_track = pickle.loads(d)
            for elem in det_track.bbox:
                # Perform image decoding and store in array
                # The two approaches should be unified
                image_as_array = cv2.imdecode(np.array(elem).astype(np.uint8), cv2.IMREAD_COLOR)

                
                imdata = Image.fromarray(image_as_array)
                LOGGER.debug(f'Performing feature extraction for received image')
                input = torch.unsqueeze(self.transform(imdata), 0)
                input = input.to(self.device)

                with FTimer(f"feature_extraction"):
                    with torch.no_grad():
                        query_feat = self.model(input)
                        LOGGER.debug(f"Extracted tensor with features: {query_feat}")

                LOGGER.debug(f"Input image size: {image_as_array.nbytes}")
                LOGGER.debug(f"Output tensor size {sys.getsizeof(query_feat.storage())}")
                total_data+=sys.getsizeof(query_feat.storage())

                # It returns a tensor, it should be transformed into a list before TX
                # result.append(self.convert_to_list(query_feat, camera_code, det_time))
                det_track.features.append(query_feat)
  

            LOGGER.info(f"Extracted ReID features for {len(det_track.bbox)} object/s received from camera {det_track.camera[0]}")
            self.write_to_fluentd(total_data)

        return pickle.dumps(det_track)        


    def extract_target_features(self, dd):
        total_data = 0

        dd = pickle.loads(dd)

        try:
            image_as_array = dd.bbox[0].astype(np.uint8)
                
            imdata = Image.fromarray(image_as_array)
            LOGGER.info(f'Performing feature extraction for target image')
            input = torch.unsqueeze(self.transform(imdata), 0)
            input = input.to(self.device)

            with FTimer(f"feature_extraction"):
                with torch.no_grad():
                    query_feat = self.model(input)
                    LOGGER.debug(f"Extracted tensor with features: {query_feat}")

            LOGGER.debug(f"Input image size: {image_as_array.nbytes}")
            LOGGER.debug(f"Output tensor size {sys.getsizeof(query_feat.storage())}")
            total_data+=sys.getsizeof(query_feat.storage())

            # It returns a tensor, it should be transformed into a list before TX
            LOGGER.info("Sending to the ReID module the target's features.")

            dd.features.append(query_feat)
            
            # result.append([query_feat.numpy().tolist(), dd.camera[0], dd.detection_time[0], dd.is_target])
            self.write_to_fluentd(total_data)

        except Exception as ex:
            LOGGER.error(f"Target's feature extraction failed {ex}")
            self.reset_op_mode()
        
        return pickle.dumps(dd)   


    def reset_op_mode(self):
        LOGGER.info("Performing operational mode emergency reset!")
        self.op_mode = "detection"
        self.target = None

    def predict(self, data, **kwargs):
        if len(data) == 0:
            return []

        if kwargs.get("new_target", False):
            return self.extract_target_features(data)
        else:
            return self.extract_features(data)

# Starting the FE module
inference_instance = FEService(estimator=Estimator)
inference_instance.start()

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


import json
import pickle
import os
import cv2

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from sedna.common.config import Context
from sedna.common.benchmark import FTimer, FluentdHelper
from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking.multi_edge_tracking import FE_ReIDService
from sedna.algorithms.reid.mAP import cosine_similarity
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult

os.environ['BACKEND_TYPE'] = 'TORCH'

image_size = Context.get_parameters('input_shape')
log_dir = Context.get_parameters('log_dir')
img_dir =  Context.get_parameters('img_dir')
gfeats = Context.get_parameters('gfeats')
qfeats = Context.get_parameters('qfeats')
imgpath = Context.get_parameters('imgpath')
dataset = Context.get_parameters('dataset')

class Estimator(FluentdHelper):

    def __init__(self, **kwargs):
        super(Estimator, self).__init__()
        LOGGER.info(f"Starting feature extraction module")

        # FE parameters
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = [int(image_size.split(",")[0]), int(image_size.split(",")[1])] 
        
        # ReID parameters
        self.log_dir = log_dir
        self.gallery_feats = torch.load(os.path.join(self.log_dir, dataset, gfeats), map_location=self.device)
        self.img_path = np.load(os.path.join(self.log_dir, dataset, imgpath))
        LOGGER.debug(f'[{self.gallery_feats.shape}, {len(self.img_path)}]')

        self.target = None
        self.target_ID = "0000"

        LOGGER.debug(f"Expected image format is {self.image_size}")

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _extract_id(self, text):
        return text.split("/")[-1].split(".")[0].split("_")[0]

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
                "worker": "cloud-feature-extractor-reid",
                "outbound_data": len(pickle.dumps(data))
            }
            
            self.send_json_msg(msg)
        except Exception as ex:
            LOGGER.error(f"Error while transmitting data to fluentd. Details: [{ex}]")

    def reid(self, query_feat, camera_code, det_time):
        LOGGER.debug(f"Running the cosine similarity function on input data")
        LOGGER.debug(f"{query_feat.shape} - {self.gallery_feats.shape}")

        with FTimer("cosine_similarity"):
            dist_mat = cosine_similarity(query_feat, self.gallery_feats)
        
        indices = np.argsort(dist_mat, axis=1)
        
        closest_match = self._extract_id(self.img_path[indices[0][0]])
        
        # Uncomment this line if you have the bboxes images available (img_dir) to create the top-10 result collage.
        # self.topK(indices, camid='mixed', top_k=10)
        result = {
            "object_id": closest_match,
            "detection_area": camera_code,
            "detection_time": det_time
        }

        # 0000 represents an unrecognized entity
        if closest_match != "0000":
            return result
        
        return None

    def load_target(self, det_track : DetTrackResult):        
        query_feat = det_track.features[0]
        query_feat = query_feat.float()

        result = self.reid(query_feat, 0, 0)

        self.target = query_feat
        self.target_ID = result.get('object_id')

        LOGGER.info(f"Target with ID {self.target_ID} acquired!")

    def create_result(self, idx, dt: DetTrackResult, id):
        return DetTrackResult(
            bbox=[dt.bbox[idx]],
            scene=dt.scene,
            confidence=[dt.confidence[idx]],
            detection_time=[dt.detection_time[idx]],
            camera=[dt.camera[idx]],
            bbox_coord=[dt.bbox_coord[idx]],
            ID=[id]
        )

    def extract_features(self, data):
        det_track = data[0]
        # det_track = pickle.loads(d)
        try:
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
                        
                det_track.features.append(query_feat)


            LOGGER.debug(f"Extracted ReID features for {len(det_track.bbox)} object/s received from camera {det_track.camera[0]}")
            self.write_to_fluentd(det_track)
        except Exception as ex:
            LOGGER.error(f"Unable to extract features [{ex}]")
            return None

        return det_track      


    def extract_target_features(self, dd):
        try:
            imdata = Image.fromarray(dd.bbox[0])
            LOGGER.info(f'Performing feature extraction for target image')
            input = torch.unsqueeze(self.transform(imdata), 0)
            input = input.to(self.device)

            with FTimer(f"feature_extraction"):
                with torch.no_grad():
                    query_feat = self.model(input)
                    LOGGER.debug(f"Extracted tensor with features: {query_feat}")

            # LOGGER.debug(f"Input image size: {dd.bbox[0].nbytes}")
            # LOGGER.debug(f"Output tensor size {sys.getsizeof(query_feat.storage())}")
            # total_data+=sys.getsizeof(query_feat.storage())

            # It returns a tensor, it should be transformed into a list before TX
            LOGGER.info("Sending to the ReID module the target's features.")

            dd.features = [query_feat]
            dd.is_target = True
            
            # result.append([query_feat.numpy().tolist(), dd.camera[0], dd.detection_time[0], dd.is_target])
            self.write_to_fluentd(dd)

        except Exception as ex:
            LOGGER.error(f"Target's feature extraction failed {ex}")
            self.reset_op_mode()
            return None
        
        return dd   


    def reset_op_mode(self):
        LOGGER.info("Performing operational mode emergency reset!")
        self.op_mode = "detection"
        self.target = None

    def predict(self, data, **kwargs):
        if data == None:
            return None

        if kwargs.get("new_target", False):
            det_track = self.extract_target_features(data)
        else:
            det_track = self.extract_features(data)

        # We use a dictionary to keep track of the ReID objects.
        # For each object, we print at the end localization and tracking information.
        reid_dict = {}
        tresult = None

        if det_track.is_target:
            self.load_target(det_track)
            return None

        for idx, elem in enumerate(det_track.bbox):
            query_feat = det_track.features[idx]
            query_feat = query_feat.float()

            result = self.reid(query_feat, det_track.camera[idx], det_track.detection_time[idx]) 

            # Create new dettrack object and exit when the target is found (return only new instance)
            if result != None:
                if kwargs['op_mode'] == "tracking":
                    if self.target_ID == result.get('object_id'):
                        LOGGER.info(f"Target found!")
                        reid_dict[result.get('object_id')] = result
                        # det_track.ID.append(result.get('object_id'))
                        tresult = self.create_result(idx, det_track, result.get('object_id'))
                    else:
                        LOGGER.debug(f"Target {self.target_ID} not found!")
                        # det_track.ID.append(-1)
                else:
                    reid_dict[result.get('object_id')] = result
                    tresult = det_track

        for key in reid_dict:
            LOGGER.info(json.dumps(reid_dict[key]))

        return tresult

# Starting the FE module
inference_instance = FE_ReIDService(estimator=Estimator)
inference_instance.start()

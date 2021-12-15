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
import distutils

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Any, NoReturn, List

from sedna.common.config import Context
from sedna.common.benchmark import FTimer, FluentdHelper
from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking.multi_edge_tracking import FE_ReIDService
from sedna.algorithms.reid.mAP import cosine_similarity
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult
from multi_img_matching import match_query_to_targets

os.environ['BACKEND_TYPE'] = 'TORCH'

image_size = Context.get_parameters('input_shape')

# GALLERY PARAMS
log_dir = Context.get_parameters('log_dir')
img_dir = Context.get_parameters('img_dir')
gfeats = Context.get_parameters('gfeats')
qfeats = Context.get_parameters('qfeats')
imgpath = Context.get_parameters('imgpath')
dataset = Context.get_parameters('dataset')

# NO-GALLERY PARAMS
use_gallery = bool(distutils.util.strtobool(Context.get_parameters('use_gallery', "True")))
match_thresh = float(Context.get_parameters('match_thresh', 800))


class Estimator(FluentdHelper):

    def __init__(self, **kwargs):
        super(Estimator, self).__init__()
        LOGGER.info(f"Starting feature extraction module")

        # Initialize ReID feature extraction module
        self.model = None

        # Device and input parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = [int(image_size.split(",")[0]),
                           int(image_size.split(",")[1])]
        
        # Data transformation
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize query features
        self.query_feat = None
        self.match_thresh = match_thresh

        self.op_mode = None
        self.use_gallery = use_gallery

        # ReID parameters: CLARIFY
        if self.use_gallery:
            self.log_dir = log_dir
            self.gallery_feats = torch.load(os.path.join(self.log_dir, dataset, gfeats), map_location=self.device)
            self.img_path = np.load(os.path.join(self.log_dir, dataset, imgpath))
            LOGGER.debug(f'[{self.gallery_feats.shape}, {len(self.img_path)}]')
            self.target = None
            self.target_ID = "0000"

        LOGGER.debug(f"Expected image format is {self.image_size}")
    
    def load(self,
             model_url: str = "") -> None:
        """Load the pre-trained ReID weights."""

        assert os.path.isfile(model_url), FileNotFoundError("ReID model not found at {}.".format(model_url))
        self.model = torch.load(model_url, map_location=torch.device(self.device))

    def evaluate(self) -> NoReturn:
        """Turn eval mode on for the model."""

        LOGGER.debug(f"Setting ReID Feature Extraction module to eval mode.")
        self.model.eval()

    def extract_features(self, data):
        det_track = data[0]
        # det_track = pickle.loads(d)
        try:
            for elem in det_track.bbox:
                # Perform image decoding and store in array
                # The two approaches should be unified
                image_as_array = cv2.imdecode(elem, cv2.IMREAD_COLOR)

                
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


    def extract_target_features(self, new_query_info) -> Any:
        """Extract the features for the query image. This function is invoked when a new query image is provided."""
        self.query_feat = None

        LOGGER.info(f"Received {len(new_query_info.bbox)} sample images for the target.")
        new_query_info.features = []
        
        for image in new_query_info.bbox:
            # new_query_info contains the query image.
            try:
                query_img = Image.fromarray(image)
            except Exception as ex:
                LOGGER.error(f"Query image not found. Error [{ex}]")
                self.reset_op_mode()
                return None

            # Attempt forward pass
            try:
                input = torch.unsqueeze(self.transform(query_img), 0).to(self.device)
                with FTimer(f"feature_extraction"):
                    with torch.no_grad():
                        query_feat = self.model(input)
                        LOGGER.debug(f"Extracted tensor with features: {query_feat}")

                # It returns a tensor, it should be transformed into a list before TX
                new_query_info.features.append(query_feat)
                new_query_info.is_target = True

            except Exception as ex:
                LOGGER.error(f"Feature extraction failed for Query image. Error [{ex}]")
                self.reset_op_mode()
                return None

        LOGGER.info("Saving target features vector.")
        self.query_feat = new_query_info.features

        return new_query_info 

    def reid_per_frame(self, candidate_feats: torch.Tensor) -> int:
        """
        For each frame, this function receives the ReID features for all the detected boxes. The similarity is computed
        between the query features and the candidate features (from the boxes). If matching score for all detected boxes
        is less than match_thresh, the function returns None signifying that no match has been found. Else, the function
        returns the index of the candidate feature with the highest matching score.
        @param candidate_feats: ...
        @return: match_id [int] which points to the index of the matched detection.
        """

        if self.query_feat == None:
            LOGGER.warning("Target has not been set!")
            return -1

        # gfeats, img_path = inference(self.model, self.query_feat, len(self.query_feat))

        # for cf in candidate_feats:
        #     dist_mat = cosine_similarity(cf, gfeats)
        #     LOGGER.info(dist_mat)

        match_id, match_score = match_query_to_targets(self.query_feat, candidate_feats, False)
        LOGGER.info(match_score)
        if float(match_score) < self.match_thresh:
            return -1, -1

        LOGGER.info(f"Selected ID {match_id} with score {match_score}")

        return match_id, match_score


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

    def predict(self, data, **kwargs) -> Any:
        """Implements the on-the-fly ReID where detections per frame are matched with the candidate boxes."""
        tresult = None

        if data == None:
            return None

        if kwargs.get("new_target", False):
            # if a new query is provided, update query features
            det_track = self.extract_target_features(data)
        else:
            # otherwise, get features for all the detected boxes
            det_track = self.extract_features(data)

        # prepare results

        if self.use_gallery:
            reid_dict = {}

            # if prediction is called for the query feature
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
        
        else:
            # Keep record of the ReID objects
            if det_track.is_target:
                return tresult

            # get id of highest match
            match_id, match_score = self.reid_per_frame(det_track.features)

            if match_id < 0:
                return tresult

            result = {
                "match_id": str(match_score),
                "detection_area": det_track.camera[match_id],
                "detection_time": det_track.detection_time[match_id]
            }

            if kwargs["op_mode"] == "tracking":
                tresult = self.create_result(match_id, det_track, f"{match_score}")
            else:
                tresult = det_track

            LOGGER.info(result)

        return tresult

    ### SUPPORT FUNCTIONS ###

    def load_target(self, det_track : DetTrackResult):  
        # Pay attention, load_target is used only when a reid gallery is available.
        # Hence, with just use one image to do ReID, we don't need more than that.      
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

    def _extract_id(self, text):
        return text.split("/")[-1].split(".")[0].split("_")[0]

    def reset_op_mode(self):
        LOGGER.info("Performing operational mode emergency reset!")
        self.op_mode = "detection"
        self.target = None

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


# Starting the FE module
inference_instance = FE_ReIDService(estimator=Estimator)
inference_instance.start()

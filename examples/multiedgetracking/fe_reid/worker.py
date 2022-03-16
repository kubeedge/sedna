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


from functools import reduce
import json
import pickle
import os
import threading
import cv2
import distutils
import copy

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
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult, OP_MODE
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

class Target:
    def __init__(self, _userid, _features, _targetid="0000") -> None:
        self.userid : str = _userid
        self.features : List = _features
        self.targetid : str = _targetid

class Estimator(FluentdHelper):

    def __init__(self, **kwargs):
        super(Estimator, self).__init__()
        # Initialize ReID feature extraction module
        self.model = None

        # Device and input parameters
        if torch.cuda.is_available():
            self.device = "cuda"
            LOGGER.info("Using GPU")
        else:
            self.device = "cpu"
            LOGGER.info("Using CPU")

        self.image_size = [int(image_size.split(",")[0]),
                           int(image_size.split(",")[1])]
        
        # Data transformation
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Service parameters
        self.parameters = None
        self.use_gallery = use_gallery
        self.targets_list : List[Target] = []

        # ReID parameters when using gallery
        if self.use_gallery:
            LOGGER.info("Enabled ReID with gallery!")
            self.log_dir = log_dir
            self.gallery_feats = torch.load(os.path.join(self.log_dir, dataset, gfeats), self.device)
            self.img_path = np.load(os.path.join(self.log_dir, dataset, imgpath))

    
    def load(self,
             model_url: str = "") -> None:
        """Load the pre-trained ReID weights."""

        assert os.path.isfile(model_url), FileNotFoundError("ReID model not found at {}.".format(model_url))
        self.model = torch.load(model_url, map_location=torch.device(self.device))
        self.model.to(self.device)
        
    def evaluate(self) -> NoReturn:
        """Turn eval mode on for the model."""

        LOGGER.debug(f"Setting ReID Feature Extraction module to eval mode.")
        self.model.eval()

    def extract_features(self, data):
        cdata = data.copy()
        input_batch = None
        j = 0

        try:
            all_bboxes = list(map(lambda x: x.bbox, cdata)) # list of lists
            total_bboxes = reduce(lambda count, l: count + len(l), all_bboxes, 0)

            with FTimer(f"data_loading"):
                for bbox_group in all_bboxes:
                    for elem in bbox_group:
                        # Perform image decoding and store in array
                        # The two approaches should be unified
                        image_as_array = cv2.imdecode(elem, cv2.IMREAD_COLOR)
                        imdata = Image.fromarray(image_as_array)
                        LOGGER.debug(f'Performing feature extraction for received image')
                        # input = torch.unsqueeze(self.transform(imdata), 0)
                        input = self.transform(imdata)

                        if j == 0:
                            input_batch = torch.zeros(total_bboxes, input.shape[0], input.shape[1], input.shape[2], dtype=torch.float)

                        input_batch[j, :, :, :] = input
                        j += 1

                input_batch = input_batch.to(self.device)

                # do forward pass once
                qf = None
                with torch.no_grad():
                        with FTimer(f"feature_extraction_forward"):
                            query_feat = self.model(input_batch)
                            qf = query_feat.to(self.device)

                offset = 0
                for k, bbox_group in enumerate(all_bboxes):
                    data[k].features = []
                    num_person = len(bbox_group)    
                    for j in range(offset, num_person+offset):
                        f = torch.unsqueeze(qf[j, :], 0)
                        #np.expand_dims(qf[i, :], 0)
                        data[k].features.append(f)
                    offset += num_person

                LOGGER.debug(f"Extracted ReID features for {offset} object/s")
                # self.write_to_fluentd(det_track)
        except Exception as ex:
            LOGGER.error(f"Unable to extract features [{ex}]")
            return None

        return data

        # try:
        #     for idx, det_track in enumerate(cdata):
        #         # combine all images
        #         if det_track:
        #             input_batch = None
        #             with FTimer(f"data_loading"):
        #                 for i, elem in enumerate(det_track.bbox):
        #                     # Perform image decoding and store in array
        #                     # The two approaches should be unified
        #                     image_as_array = cv2.imdecode(elem, cv2.IMREAD_COLOR)
        #                     imdata = Image.fromarray(image_as_array)
        #                     LOGGER.debug(f'Performing feature extraction for received image')
        #                     # input = torch.unsqueeze(self.transform(imdata), 0)
        #                     input = self.transform(imdata)

        #                     if i == 0:
        #                         input_batch = torch.zeros(len(det_track.bbox), input.shape[0], input.shape[1], input.shape[2],
        #                                                 dtype=torch.float)
        #                     input_batch[i, :, :, :] = input

                    
        #             input_batch = input_batch.to(self.device)

        #             # do forward pass once
        #             qf = None
        #             with torch.no_grad():
        #                     with FTimer(f"feature_extraction_forward"):
        #                         query_feat = self.model(input_batch)
        #                         qf = query_feat.to(self.device)

        #             num_person = qf.shape[0]
        #             data[idx].features = []

        #             for j in range(0, num_person):
        #                 f = torch.unsqueeze(qf[j, :], 0)
        #                 #np.expand_dims(qf[i, :], 0)
        #                 data[idx].features.append(f)

        #             LOGGER.debug(f"Extracted ReID features for {len(det_track.bbox)} object/s received from camera {det_track.camera}")
        #             self.write_to_fluentd(det_track)
        # except Exception as ex:
        #     LOGGER.error(f"Unable to extract features [{ex}]")
        #     return None

        # return data

    def extract_target_features(self, ldata) -> Any:
        """Extract the features for the query image. This function is invoked when a new query image is provided."""

        # We reset the previous targets.
        # We have to do this to avoid desync in some corner case.
        self.targets_list.clear()

        try:
            for new_query_info in ldata:
                LOGGER.info(f"Received {len(new_query_info.bbox)} sample images for the target for user {new_query_info.userID}.")
                new_query_info.features = []
                
                for image in new_query_info.bbox:
                    # new_query_info contains the query image.
                    try:
                        query_img = Image.fromarray(image[:,:,:3]) #dropping non-color channels
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
                        LOGGER.error(f"Feature extraction failed for query image. Error [{ex}]")
                        self.reset_op_mode()
                        return None
                
                self.targets_list.append(Target(new_query_info.userID, new_query_info.features))
        except Exception as ex:
            LOGGER.error(f"Unable to extract features for the target [{ex}]")
        
            # self._update_targets_list(new_query_info.userID, new_query_info.features)

    def _update_targets_list(self, userid, features):
        target = list(filter(lambda x: x.userid == userid, self.targets_list))

        if len(target) > 0:
            LOGGER.info(f"Update existing features for user {userid}")
            target[0].features = features
        else:
            LOGGER.info(f"Create new list of features for user {userid}")
            self.targets_list.append(Target(userid, features))


    def reid_per_frame(self, features, candidate_feats: torch.Tensor) -> int:
        """
        For each frame, this function receives the ReID features for all the detected boxes. The similarity is computed
        between the query features and the candidate features (from the boxes). If matching score for all detected boxes
        is less than match_thresh, the function returns None signifying that no match has been found. Else, the function
        returns the index of the candidate feature with the highest matching score.
        @param candidate_feats: ...
        @return: match_id [int] which points to the index of the matched detection.
        """

        if features == None:
            LOGGER.warning("Target has not been set!")
            return -1

        # gfeats, img_path = inference(self.model, self.query_feat, len(self.query_feat))

        # for cf in candidate_feats:
        #     dist_mat = cosine_similarity(cf, gfeats)
        #     LOGGER.info(dist_mat)

        with FTimer(f"reid_no_gallery"):
            match_id, match_score = match_query_to_targets(features, candidate_feats, False)

        LOGGER.debug(f"{match_score}")

        return match_id, match_score


    def reid(self, query_feat, camera_code, det_time):
        LOGGER.debug(f"Running the cosine similarity function on input data")

        with FTimer("reid"):
            dist_mat = cosine_similarity(query_feat.to(self.device), self.gallery_feats)
        
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

    def predict(self, data, new_target=False, **kwargs) -> Any:
        """Implements the on-the-fly ReID where detections per frame are matched with the candidate boxes."""
        tresult = []

        # First of all, we update the class parameters
        self.parameters = kwargs

        # Here we dedice if we received a set of targets or a
        # normal image to process.

        if new_target:
            # if a new query is provided, update query features
            self.extract_target_features(data)

            if self.use_gallery:
                self._extract_target_id()
            return tresult

        # Extract the features from all the received DetTrack objects
        data = data[0] # This is because Kafka sends us a list of lists

        dettrack_objs_with_features = self.extract_features(data)

        LOGGER.info(f"Processing a batch of {len(dettrack_objs_with_features)} DetTrack objects enriched with features")

        for dettrack_obj in dettrack_objs_with_features:
            try:
                if self.use_gallery:
                    if dettrack_obj is not None:
                        reid_result = getattr(self, kwargs["op_mode"].value)(dettrack_obj)
                        if reid_result is not None:
                            tresult.append(reid_result)
                else:
                    reid_result = getattr(self, kwargs["op_mode"].value + "_no_gallery")(dettrack_obj)
                    if reid_result is not None:
                        tresult.append(reid_result)

            except AttributeError as ex:
                LOGGER.error(f"Error in dynamic function mapping. [{ex}]")
                return tresult

        return tresult

    ### OP_MODE FUNCTIONS ###

    def detection_no_gallery(self, det_track):
        LOGGER.warning(f"This operational mode ({self.parameters['op_mode']}) is not allowed without gallery!")
        return None

    def tracking_no_gallery(self, det_track : DetTrackResult):
        # tresult = []
        det_track.targetID = [-1] * len(det_track.bbox) 
        
        for target in self.targets_list:
            # get id of highest match for each userid
            match_id, match_score = self.reid_per_frame(target.features, det_track.features)

            if float(match_score) >= self.parameters['threshold']:
                result = {
                    "userID": str(target.userid),
                    "match_id": str(match_score),
                    "detection_area": det_track.camera,
                    "detection_time": det_track.detection_time
                }

                # tresult.append(self.create_result(match_id, det_track, f"{match_score}", target.userid))
                det_track.targetID[match_id]= str(target.userid)
                det_track.userID = target.userid
                det_track.is_target = match_id

                LOGGER.info(result)
        
        if det_track.targetID.count(-1) == len(det_track.targetID):
            # No target found, we don't send any result back
            return None

        # Moving tensors to CPU (we don't need them on GPU anymore)
        det_track.features = [t.to('cpu') for t in det_track.features]

        return det_track

    def detection(self, det_track :DetTrackResult):
        reid_dict = {}
        
        for idx, _ in enumerate(det_track.bbox):
            query_feat = det_track.features[idx].float()

            result = self.reid(query_feat, det_track.camera, det_track.detection_time) 

            # Create and append a new dettrack object and exit when the target is found
            if result != None:
                reid_dict[result.get('object_id')] = result
                det_track.targetID.append(result.get('object_id'))
                det_track.userID = "DEFAULT"

        # Moving tensors to CPU (we don't need them on GPU anymore)
        det_track.features = [t.to('cpu') for t in det_track.features]

        for key in reid_dict:
            LOGGER.info(json.dumps(reid_dict[key]))

        # Moving tensors to CPU (we don't need them on GPU anymore)
        det_track.features = [t.to('cpu') for t in det_track.features]

        return det_track

    def tracking(self, det_track : DetTrackResult):
        reid_dict = {}

        for idx, _ in enumerate(det_track.bbox):
            query_feat = det_track.features[idx].float()

            result = self.reid(query_feat, det_track.camera, det_track.detection_time) 

            # Edit in-place the dettrack object
            if result != None:
                for target in self.targets_list:
                    if target.targetid == result.get('object_id'):
                        LOGGER.info(f"Target {target.targetid} found!")
                        reid_dict[result.get('object_id')] = result
                        det_track.targetID.append(result.get('object_id'))
                        det_track.userID = target.userid
                    else:
                        LOGGER.debug(f"Target {target.targetid} not found!")
                        det_track.targetID.append(-1)

        # Moving tensors to CPU (we don't need them on GPU anymore)
        det_track.features = [t.to('cpu') for t in det_track.features]

        for key in reid_dict:
            LOGGER.info(json.dumps(reid_dict[key]))

        return self._filter_targets(det_track)

    ### SUPPORT FUNCTIONS ###

    def _filter_targets(self, det_track : DetTrackResult):
        for idx, elem in enumerate(det_track.targetID):
            if elem != -1:
                item = DetTrackResult(
                    bbox=[det_track.bbox[idx]],
                    scene=det_track.scene,
                    confidence=[det_track.confidence[idx]],
                    detection_time=det_track.detection_time,
                    camera=det_track.camera,
                    bbox_coord=[det_track.bbox_coord[idx]],
                    ID=[elem]
                )
                item.userID = det_track.userID
                return item

        return []        

    def _extract_target_id(self):  
        # Pay attention, load_target is used only when a reid gallery is available.
        # Hence, with just use one image to do ReID, we don't need more than that.      
        for target in self.targets_list:
            result = self.reid(target.features[0].float(), 0, 0)
            target.targetid = result.get('object_id')

            LOGGER.info(f"Target with ID {target.targetid} acquired!")

    def create_result(self, idx, dt: DetTrackResult, id, userid):
        item = DetTrackResult(
            bbox=[dt.bbox[idx]],
            scene=dt.scene,
            confidence=[dt.confidence[idx]],
            detection_time=dt.detection_time,
            camera=dt.camera,
            bbox_coord=[dt.bbox_coord[idx]],
            ID=[id]
        )
        item.userID = userid
        return item

    def _extract_id(self, text):
        return text.split("/")[-1].split(".")[0].split("_")[0]

    def reset_op_mode(self):
        LOGGER.info("Performing operational mode emergency reset!")
        self.parameters['op_mode'] = OP_MODE.DETECTION
        self.targets_list = []

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
# Parallel processing should not be used currently as the code logic
# doesn't support it yet.
worker_pool = [FE_ReIDService(estimator=Estimator)] * 1
for worker in worker_pool:
    threading.Thread(target=worker.start, daemon=False).start()

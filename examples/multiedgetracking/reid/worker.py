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

import distutils

import cv2
import numpy as np
import os
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult
import torch, json
from PIL import Image

from sedna.core.multi_edge_tracking import ReIDService
from sedna.common.config import Context
from sedna.common.log import LOGGER
from sedna.common.benchmark import FTimer
from sedna.algorithms.reid.mAP import cosine_similarity

os.environ['BACKEND_TYPE'] = 'TORCH'

# GALLERY PARAMS
log_dir = Context.get_parameters('log_dir')
img_dir = Context.get_parameters('img_dir')
gfeats = Context.get_parameters('gfeats')
qfeats = Context.get_parameters('qfeats')
imgpath = Context.get_parameters('imgpath')
dataset = Context.get_parameters('dataset')

# NO-GALLERY PARAMS
use_gallery = bool(distutils.util.strtobool(Context.get_parameters('use_gallery', "True")))
match_thresh = Context.get_parameters('match_thresh', 1.2)


class Estimator():

    def __init__(self, **kwargs):
        LOGGER.info("Starting ReID module")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize query features
        self.query_feat = None
        self.match_thresh = match_thresh

        self.use_gallery = use_gallery

        # ReID parameters
        if self.use_gallery:
            self.log_dir = log_dir
            self.gallery_feats = torch.load(os.path.join(self.log_dir, dataset, gfeats), map_location=self.device)
            self.img_path = np.load(os.path.join(self.log_dir, dataset, imgpath))
            LOGGER.debug(f'[{self.gallery_feats.shape}, {len(self.img_path)}]')
            self.target = None
            self.target_ID = "0000"

    def _extract_id(self, text):
        return text.split("/")[-1].split(".")[0].split("_")[0]

    def _write_id_on_image(self, img, text):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        textX = (img.shape[1] - textsize[0]) / 2
        textY = (img.shape[0] + textsize[1]) / 2

        # add text centered on image
        cv2.putText(img, text, (int(textX), int(textY) ), font, 1, (255, 255, 255), 2)


    def load(self, model_name="", **kwargs):
        pass

    def reid_per_frame(self, candidate_feats: torch.Tensor) -> int:
        """
        For each frame, this function receives the ReID features for all the detected boxes. The similarity is computed
        between the query features and the candidate features (from the boxes). If matching score for all detected boxes
        is less than match_thresh, the function returns None signifying that no match has been found. Else, the function
        returns the index of the candidate feature with the highest matching score.
        @param candidate_feats: ...
        @return: match_id [int] which points to the index of the matched detection.
        """

        candidate_feats = torch.stack(candidate_feats, 1)[0, :, :] 

        # Compute distance between
        sim_measure = cosine_similarity(self.query_feat, candidate_feats)[0]

        match_score = np.sort(sim_measure.flatten(), axis=0)[0]
        if isinstance(match_score, list):
            match_score = np.array(match_score, dtype=np.float)

        if not np.any(match_score < self.match_thresh):
            return -1

        # Return id corresponding to the highest match
        match_id = np.argsort(sim_measure.flatten(), axis=0)[-1]

        return match_id

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
        LOGGER.debug(query_feat)

        result = self.reid(query_feat, 0, 0)

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

    def predict(self, data, **kwargs):
        det_track = data
        tresult = None

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

            # Pool features from all candidate boxes
            candidate_feats = [None] * len(det_track.bbox)
            for idx, elem in enumerate(det_track.bbox):
                candidate_feats[idx] = det_track.bbox[idx]

            # get id of highest match
            match_id = self.reid_per_frame(candidate_feats)

            if match_id < 0:
                return tresult

            result = {
                "match_id": match_id,
                "detection_area": det_track.camera[match_id],
                "detection_time": det_track.detection_time[match_id]
            }

            if kwargs["op_mode"] == "tracking":
                tresult = self.create_result(match_id, det_track, f"{match_id}")
            else:
                tresult = det_track

            LOGGER.info(result)

        return tresult

    def topK(self, indices, camid, top_k=10, img_size=[128, 128]):
        LOGGER.debug("Saving top-10 results")
        figure = None
        for k in range(top_k):
            img = Image.open(os.path.join(img_dir, self.img_path[indices[0][k]])).resize(
                (img_size[1], img_size[0]))
            img = np.asarray(img)
            self._write_id_on_image(img, self._extract_id(self.img_path[indices[0][k]]))
            if figure is not None:
                figure = np.hstack((figure, img))
            else:
                figure = img
        figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        result_path = os.path.join(self.log_dir, "results")

        if not os.path.exists(result_path):
            LOGGER.debug('Creating a new folder named results in {}'.format(self.log_dir))
            os.makedirs(result_path)

        cv2.imwrite(os.path.join(
            # result_path, "{}-cam{}-{}.png".format(indices[0][0], camid, time.time())), figure)
            result_path, "{}-cam{}.png".format(indices[0][0], camid)), figure)

# Starting the ReID module
inference_instance = ReIDService(estimator=Estimator)
inference_instance.start()

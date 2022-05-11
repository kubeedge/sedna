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

from distutils import util
import time
import torch
import numpy as np
from PIL import Image
from threading import Thread
from typing import List
from store_result import save_image
from sedna.algorithms.reid.multi_img_matching import match_query_to_targets
from sedna.algorithms.reid.close_contact_estimation import ContactTracker
from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking.components.reid import ReID
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult, OP_MODE, Target
from sedna.core.multi_edge_tracking.utils import get_parameters
from sedna.datasources.obs.connector import OBSClientWrapper

class ReIDWorker():

    def __init__(self, **kwargs):       
        # Service parameters
        self.op_mode = OP_MODE(get_parameters('op_mode', 'covid19')) 
        self.threshold = get_parameters('match_threshold', 0.75)
        self.user_id = get_parameters('user_id', "DEFAULT")
        self.query_images = str(get_parameters('query_images', "/data/query/sample.png")).split("|")

        self.target = None
        self.targets_list : List[Target] = []

        self.results_base_folder = "/data/images/"

        self.CT = ContactTracker(draw_top_view=False)
        
        self.enable_obs =  bool(util.strtobool(get_parameters('ENABLE_OBS', "False")))

        if self.enable_obs:
            self.obs_client = OBSClientWrapper(app_token=get_parameters('OBS_TOKEN', ''))

        super(ReIDWorker, self).__init__()
    
    def update_plugin(self, status):
        # Update target
        if self.op_mode != OP_MODE.DETECTION:
            LOGGER.info("Loading target query images")

            # The target collection is a list of targets/userid that might grow overtime
            img_arr = []
            for image in self.query_images:
                img_arr.append(np.asarray(Image.open(image)))
                
            data = DetTrackResult(0, img_arr, None, [], 0, 0)
            data.userID = self.user_id

        return [data]

    def update_target(self, ldata):
        LOGGER.info(f"Target updated for user {ldata[0].userid} with {len(ldata[0].features)} feature vectors!")
        self.targets_list = ldata 

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

        match_id, match_score = match_query_to_targets(features, candidate_feats, False)
        return match_id, match_score


    def predict(self, data, **kwargs):
        """Implements the on-the-fly ReID where detections per frame are matched with the candidate boxes."""
        tresult = []

        for dettrack_obj in data:
            try:
                reid_result = getattr(self, self.op_mode.value + "_no_gallery")(dettrack_obj)
                if reid_result is not None:
                    tresult.append(reid_result)
                    self.store_result(reid_result)
            except AttributeError as ex:
                LOGGER.error(f"Error in dynamic function mapping. [{ex}]")
                return tresult

        return tresult

    ### OP_MODE FUNCTIONS ###

    def covid19_no_gallery(self, det_track):
        return self.tracking_no_gallery(det_track)

    def detection_no_gallery(self, det_track):
        LOGGER.warning(f"This operational mode ({self.op_mode}) is not supported without gallery.")
        return None

    def tracking_no_gallery(self, det_track : DetTrackResult):
        det_track.targetID = [-1] * len(det_track.bbox_coord) 
        
        for target in self.targets_list:
            # get id of highest match for each userid
            match_id, match_score = self.reid_per_frame(target.features, det_track.features)
            
            result = {
                "userID": str(target.userid),
                "match_id": str(match_score),
                "detection_area": det_track.camera,
                "detection_time": det_track.detection_time
            }

            if float(match_score) >= self.threshold:
                det_track.targetID[match_id]= str(target.userid)
                det_track.userID = target.userid
                det_track.is_target = match_id

            LOGGER.info(result)
        
        if det_track.targetID.count(-1) == len(det_track.targetID):
            # No target found, we don't send any result back
            return None

        return det_track

    def store_result(self, det_track : DetTrackResult):
        try:
            filename = save_image(det_track, self.CT, folder=f"{self.results_base_folder}{det_track.userID}/")
            if self.enable_obs:
                self.obs_client.upload_file(f"{self.results_base_folder}{det_track.userID}/", filename, f"/media/reid/{det_track.userID}")
        except Exception as ex:
            LOGGER.error(f"Unable to save image: {ex}") 

class Bootstrapper(Thread):
    def __init__(self):
        super().__init__()

        self.daemon = True
        self.folder = "/data/"
        self.retry = 3
        self.job = ReID(models=[ReIDWorker()], asynchronous=False)

    def run(self) -> None:
        LOGGER.info("Loading data from disk.")

        while self.retry > 0:
            files = self.job.get_files_list(self.folder)

            if files:
                LOGGER.debug(f"Loaded {len(files)} files.")
                for filename in files:
                    data = self.job.read_from_disk(filename)
                    if data:
                        LOGGER.debug(f"File {filename} loaded!")
                        self.job.put(data)
                        self.job.delete_from_disk(filename)
                break
            else:
                LOGGER.warning("No data available to process!")
                self.retry-=1
                time.sleep(5)

        LOGGER.info("ReID job completed.")

# Start the ReID job.
if __name__ == '__main__':
    bs = Bootstrapper()
    bs.run()
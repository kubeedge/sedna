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

import cv2
import numpy as np
import os
import torch, json
from PIL import Image

from sedna.core.multi_edge_tracking import ReIDService
from sedna.common.config import Context
from sedna.common.log import LOGGER
from sedna.common.benchmark import FTimer
from sedna.algorithms.reid.mAP import cosine_similarity

os.environ['BACKEND_TYPE'] = 'TORCH'

log_dir = Context.get_parameters('log_dir')
img_dir =  Context.get_parameters('img_dir')
gfeats = Context.get_parameters('gfeats')
qfeats = Context.get_parameters('qfeats')
imgpath = Context.get_parameters('imgpath')
dataset = Context.get_parameters('dataset')

class Estimator:

    def __init__(self, **kwargs):
        LOGGER.info("Starting ReID module")

        self.log_dir = log_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gallery_feats = torch.load(os.path.join(self.log_dir, dataset, gfeats), map_location=self.device)
        self.img_path = np.load(os.path.join(self.log_dir, dataset, imgpath))
        LOGGER.debug(f'[{self.gallery_feats.shape}, {len(self.img_path)}]')

    def _extract_id(self, text):
        return text.split("/")[-1].split(".")[0].split("_")[0]

    def _write_id_on_image(self, img, text):
        # setup text
        font = cv2.FONT_HERSHEY_SIMPLEX
        #print(text)
        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        textX = (img.shape[1] - textsize[0]) / 2
        textY = (img.shape[0] + textsize[1]) / 2

        # add text centered on image
        cv2.putText(img, text, (int(textX), int(textY) ), font, 1, (255, 255, 255), 2)


    def load(self, model_name="", **kwargs):
        pass

    def predict(self, data, **kwargs):
        # We use a dictionary to keep track of the ReID objects.
        # For each object, we print at the end localization and tracking information.
        reid_dict = {}

        for d in data:
            for dd in d:
                temp = np.array(dd[0])
                camera_code = dd[1]
                det_time = dd[2] 

                query_feat = torch.from_numpy(temp)
                query_feat = query_feat.float()
                
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
                    reid_dict[self._extract_id(self.img_path[indices[0][0]])] = result
                
        for key in reid_dict:
            LOGGER.info(json.dumps(reid_dict[key]))
            
        self.save_result(reid_dict)
            # LOGGER.info(f"Container with ID {self._extract_id(self.img_path[indices[0][0]])} detected in area {camera_code} with timestamp {det_time}")
        
        self.save_result(reid_dict)

        return indices[0][:]

    def save_result(self, data):
        # Not implemented. Options are:
        # 1. Save on disk (bad)
        # 2. Save in-memory (bad)
        # 3. Write to Kafka/DB (good)
        pass

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
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
import torch

from utils.getter import *
from PIL import Image

from sedna.core.multi_edge_tracking import ReIDService
from sedna.common.config import Context
from sedna.common.log import LOGGER
from sedna.common.benchmark import FTimer

os.environ['BACKEND_TYPE'] = 'TORCH'

log_dir = Context.get_parameters('reid_log_dir')
gfeats = Context.get_parameters('gfeats')
qfeats = Context.get_parameters('qfeats')
imgpath = Context.get_parameters('imgpath')

class Estimator:

    def __init__(self, **kwargs):
        LOGGER.info("Initializing cloud ReID worker ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log_dir = log_dir
        self.gallery_feats = torch.load(os.path.join(self.log_dir, gfeats), map_location=self.device )
        self.img_path = np.load(os.path.join(self.log_dir, imgpath))

        LOGGER.info(f'[{self.gallery_feats.shape}, {len(self.img_path)}]')


    def predict(self, data, **kwargs):
        temp = np.array(data)
        query_feat = torch.from_numpy(temp)
        query_feat = query_feat.float()
        
        LOGGER.info(f"Running the cosine similarity function on input data")
        LOGGER.info(f"{query_feat.shape} - {self.gallery_feats.shape}")
        with FTimer("cosine_similarity"):
            dist_mat = cosine_similarity(query_feat, self.gallery_feats)
            indices = np.argsort(dist_mat, axis=1)
        
        self.save_result(indices, camid='mixed', top_k=10)

        return indices[0][:]

    def save_result(self, indices, camid, top_k=10, img_size=[128, 128]):
        LOGGER.info("Saving top-10 results")
        figure = None
        for k in range(top_k):
            img = np.asarray(Image.open(os.path.join("/code/ai_models/deep_eff_reid/", self.img_path[indices[0][k]])).resize(
                (img_size[1], img_size[0])))
            if figure is not None:
                figure = np.hstack((figure, img))
            else:
                figure = img
        figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        result_path = os.path.join(self.log_dir, "results")

        if not os.path.exists(result_path):
            LOGGER.info('Creating a new folder named results in {}'.format(self.log_dir))
            os.makedirs(result_path)

        cv2.imwrite(os.path.join(
            result_path, "{}-cam{}.png".format("test", camid)), figure)


# Starting the ReID module
inference_instance = ReIDService(estimator=Estimator)
inference_instance.start()
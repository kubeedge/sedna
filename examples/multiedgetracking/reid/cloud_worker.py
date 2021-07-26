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
import sys
import argparse
import random
import time
import torch

from utils.getter import *
from utils.logger import setup_logger
from PIL import Image

from sedna.core.multi_edge_tracking import ReIDService
from sedna.common.config import Context

LOG = logging.getLogger(__name__)
os.environ['BACKEND_TYPE'] = 'TORCH'

log_dir = Context.get_parameters('reid_log_dir')
gfeats = Context.get_parameters('gfeats')
qfeats = Context.get_parameters('qfeats')
imgpath = Context.get_parameters('imgpath')
image_size = Context.get_parameters('input_shape')

sys.path.append('.')

class Estimator:

    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log_dir = log_dir
        self.gallery_feats = torch.load(os.path.join(self.log_dir, gfeats), map_location=self.device )
        self.img_path = np.load(os.path.join(self.log_dir, imgpath))
        self.image_size = [image_size.split(",")[0], image_size.split(",")[1]] 

        print('[gallery_feats.shape, len(img_path)]: ', self.gallery_feats.shape, len(self.img_path))


    def predict(self, query_img, **kwargs):
        start = time.time()
        dist_mat = cosine_similarity(self.query_feat, self.gallery_feats)
        indices = np.argsort(dist_mat, axis=1)
        end = time.time()
        self.logger.info(f"cosine_similarity calc time: {end - start}")
        
        self.save_result(query_img, indices, camid='mixed', top_k=10,
                img_size=self.image_size)

    def save_result(self, test_img, indices, camid, top_k=10, img_size=[128, 128]):
        figure = np.asarray(self.query_img.resize((img_size[1], img_size[0])))
        for k in range(top_k):
            name = str(indices[0][k]).zfill(6)
            img = np.asarray(Image.open(self.img_path[indices[0][k]]).resize(
                (img_size[1], img_size[0])))
            figure = np.hstack((figure, img))
            title = name
            print(title)
        figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        result_path = os.path.join(self.log_dir, "results")

        if not os.path.exists(result_path):
            print('Create a new folder named results in {}'.format(self.log_dir))
            os.makedirs(result_path)

        cv2.imwrite(os.path.join(
            result_path, "{}-cam{}.png".format(test_img, camid)), figure)


# Starting the ReID module
inference_instance = ReIDService(estimator=Estimator)
inference_instance.start()
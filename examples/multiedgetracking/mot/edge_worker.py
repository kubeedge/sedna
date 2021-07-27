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

import numpy as np
import os
import torch
import time
import torchvision.transforms as T

from utils.getter import *
from torch.backends import cudnn

from sedna.backend.torch.nn import Backbone
from sedna.common.config import Context

LOG = logging.getLogger(__name__)
os.environ['BACKEND_TYPE'] = 'TORCH'

model_weights = Context.get_parameters('edge_model_weights')
model_name = Context.get_parameters('model_name')
image_size = Context.get_parameters('input_shape')

class Estimator:

    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = [image_size.split(",")[0], image_size.split(",")[1]] 
        cudnn.benchmark = True

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load(self, model_url=""):
        # The model should be provided by a CRD
        self.model = Backbone(num_classes=255, model_name=model_name)

    def load_weights(self):
        # Here we load the model weights from the attached volume (.yaml)
        self.model.load_param(model_weights)
        self.model = self.model.to(self.device)

    def evaluate(self):
        return self.model.eval()

    def predict(self, query_img, **kwargs):      
        LOG.info('Finding ID {} ...'.format(query_img))
        input = torch.unsqueeze(self.transform(query_img), 0)
        input = input.to(self.device)

        start = time.time()
        with torch.no_grad():
            query_feat = self.model(input)
        end = time.time()
        LOG.info(f"Feature extraction from query image: {end - start}")

        # It returns a tensor, it should be transformed into an array before TX
        return query_feat

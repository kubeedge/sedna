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

import os
import time
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List

from functools import reduce
from threading import Thread

from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking.utils import get_parameters
from sedna.core.multi_edge_tracking.plugins import PluggableModel
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult, Target
from sedna.core.multi_edge_tracking.components.feature_extraction import FEService

os.environ['BACKEND_TYPE'] = 'TORCH'

class FeatureExtractionAI(PluggableModel):

    def __init__(self, **kwargs):
        # Initialize feature extraction module
        self.model = None

        # Device and input parameters
        if torch.cuda.is_available():
            self.device = "cuda"
            LOGGER.info("Using GPU")
        else:
            self.device = "cpu"
            LOGGER.info("Using CPU")

        image_size = get_parameters('input_shape')
        self.image_size = [int(image_size.split(",")[0]),
                           int(image_size.split(",")[1])]
        
        # Data transformation
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        super(FeatureExtractionAI, self).__init__()

    
    def load(self):
        """Load the pre-trained FE weights."""
        if not self.model:
            assert os.path.isfile(self.model_path), FileNotFoundError("FE model not found at {}.".format(self.model_path))
            self.model = torch.load(self.model_path, map_location=torch.device(self.device))
            self.model.to(self.device)
        
    def evaluate(self):
        """Turn eval mode on for the model."""

        LOGGER.debug(f"Setting Feature Extraction module to eval mode.")
        self.model.eval()

    def extract_features(self, data : List[DetTrackResult] ):
        input_batch = None
        j = 0
        offset = 0

        try:
            all_bboxes = list(map(lambda x: x.bbox_coord, data)) # list of lists
            total_bboxes = reduce(lambda count, l: count + len(l), all_bboxes, 0)

            # Prepare the batch
            for idx, bbox_group in enumerate(all_bboxes):

                # Decode the original image
                imdata = cv2.imdecode(data[idx].scene, cv2.IMREAD_COLOR)
                
                for elem in bbox_group:
                    # Using the bbox coordinates we crop the original image
                    x0, y0, x1, y1 = int(elem[0]), int(elem[1]), int(elem[2]), int(elem[3])
                    crop = Image.fromarray(imdata[y0: y1, x0: x1])
                    
                    LOGGER.debug(f'Performing feature extraction for received image.')
                    input = self.transform(crop)

                    if j == 0:
                        if self.device == "cuda":
                            # initialized directly on GPU
                            input_batch = torch.cuda.FloatTensor(total_bboxes, input.shape[0], input.shape[1], input.shape[2])
                        else:
                            input_batch = torch.zeros(total_bboxes, input.shape[0], input.shape[1], input.shape[2], dtype=torch.float)

                    input_batch[j, :, :, :] = input.to(self.device)
                    j += 1

            # do forward pass once
            with torch.no_grad():
                query_feat = self.model(input_batch)
                qf = query_feat.to(self.device)

            qf = qf.to('cpu')

            # Enrich DetTrackResult object with the extracted features
            for k, bbox_group in enumerate(all_bboxes):
                num_person = len(bbox_group)    
                for j in range(offset, num_person+offset):
                    f = torch.unsqueeze(qf[j, :], 0)
                    #np.expand_dims(qf[i, :], 0)
                    data[k].features.append(f)
                offset += num_person

            LOGGER.debug(f"Extracted features for {offset} object/s.")

        except Exception as ex:
            LOGGER.error(f"Unable to extract features [{ex}].")
            return None

        return data

    def update_plugin(self, update_object, **kwargs):
        pass


    def get_target_features(self, ldata):
        """Extract the features for the query image. This function is invoked when a new query image is provided."""

        try:
            for new_query_info in ldata:
                LOGGER.info(f"Received {len(new_query_info.bbox)} sample images for the target for user {new_query_info.userID}.")
                new_query_info.features = []
                
                for image in new_query_info.bbox:
                    # new_query_info contains the query image.
                    try:
                        query_img = Image.fromarray(image[:,:,:3]) #dropping non-color channels
                    except Exception as ex:
                        LOGGER.error(f"Query image not found. Error [{ex}].")
                        return None

                    # Attempt forward pass
                    try:
                        input = torch.unsqueeze(self.transform(query_img), 0).to(self.device)
                        with torch.no_grad():
                            query_feat = self.model(input)
                            LOGGER.debug(f"Extracted tensor with features: {query_feat}.")

                        query_feat = query_feat.to('cpu')

                        # It returns a tensor, it should be transformed into a list before TX
                        new_query_info.features.append(query_feat)
                        new_query_info.is_target = True

                    except Exception as ex:
                        LOGGER.error(f"Feature extraction failed for query image. Error [{ex}].")
                        return None
                
                return [Target(new_query_info.userID, new_query_info.features)]

        except Exception as ex:
            LOGGER.error(f"Unable to extract features for the target [{ex}].")
        
        return None

    def predict(self, data, **kwargs):
        """Implements enrichment of DetTrack object with features for ReID."""

        dettrack_objs_with_features = self.extract_features(data)

        if dettrack_objs_with_features:
            return dettrack_objs_with_features
        
        return []

# we need this otherwise the program exits after creating the service
# this is a temporary solution
class Bootstrapper(Thread):
    def __init__(self):
        super().__init__()

        self.daemon = True
        FEService(models=[FeatureExtractionAI()], asynchronous=False)
               

    def run(self) -> None:
        while True:
            time.sleep(0.1)


# Starting the FE service.
if __name__ == '__main__':
    bs = Bootstrapper()
    bs.run()
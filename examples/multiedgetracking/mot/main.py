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

import time
import cv2
import logging
import numpy as np

from sedna.common.config import Context
from sedna.core.multi_edge_tracking import MultiObjectTracking

from edge_worker import Estimator

LOG = logging.getLogger(__name__)

camera_address = Context.get_parameters('video_url')

def main():
    edge_worker = MultiObjectTracking(estimator=Estimator)

    camera = cv2.VideoCapture(camera_address)
    fps = 10
    nframe = 0
    while 1:
        ret, input_yuv = camera.read()
        if not ret:
            LOG.info(
                f"camera is not open, camera_address={camera_address},"
                f" sleep 5 second.")
            time.sleep(5)
            camera = cv2.VideoCapture(camera_address)
            continue

        if nframe % fps:
            nframe += 1
            continue

        img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_BGR2RGB)
        nframe += 1
        LOG.info(f"camera is open, current frame index is {nframe}")
        edge_worker.inference(img_rgb)

if __name__ == '__main__':
    main()

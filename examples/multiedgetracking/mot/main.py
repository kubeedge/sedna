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

from sedna.common.config import Context
from sedna.core.multi_edge_tracking import MultiObjectTracking
from sedna.common.log import LOGGER
from edge_worker import Estimator

camera_address = Context.get_parameters('video_url')

def main():
    edge_worker = MultiObjectTracking(estimator=Estimator)

    camera = cv2.VideoCapture(camera_address)
    fps = 10
    nframe = 0
    
    while True:
        try:
            ret, input_yuv = camera.read()
        except Exception:
            pass

        if not ret:
            LOGGER.info(
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
        LOGGER.info(f"camera is open, current frame index is {nframe}")
        edge_worker.inference(img_rgb)

if __name__ == '__main__':
    main()

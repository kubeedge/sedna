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

import requests
import cv2
import threading

from sedna.common.config import Context
from sedna.core.multi_edge_tracking import ObjectDetector
from sedna.common.log import LOGGER
from edge_worker import Estimator

camera_address = Context.get_parameters('video_url')
stream_dispatcher = Context.get_parameters('stream_dispatcher_url')

def retrieve_rtsp_stream() -> str:
    LOGGER.debug(f'Finding target RTSP stream ...')
    if stream_dispatcher != None:
        try:
            rtsp_stream = requests.get(stream_dispatcher)
            LOGGER.debug(f'Retrieved RTSP stream with address {rtsp_stream}')
            # This is crazy, but we have to do it otherwise cv2 will silenty fail and never open the RTSP stream
            cv2_cleaned_string = rtsp_stream.text.strip().replace('"', '')
            return cv2_cleaned_string
        except Exception as ex:
            LOGGER.error(f'Unable to access stream dispatcher server, using fallback value. [{ex}]')
            return camera_address
    else:
        LOGGER.debug(f'Using RTSP from env variable with address {camera_address}')
        return camera_address
    



def start_stream_acquisition(stream_address):
    camera_code = stream_address.split("/")[-1] # WARNING: Only for demo purposes!
    edge_worker = ObjectDetector(estimator=Estimator(camera_code=camera_code))

    camera = cv2.VideoCapture(stream_address)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    fps = 0.5
    nframe = 0
    startTime = time.time()

    while True:
        try:
            # We use grab() to avoid populating the camera buffer with images from the past.
            # If the detection would be fast enough, we would not need this workaround.
            # Also, this should be event based rather than using time intervals.
            ret = camera.grab()
            nowTime = time.time()
        except Exception as ex:
            LOGGER.error(f'Unable to access stream [{ex}]')
    
        if not ret:
            LOGGER.debug(
                f" Camera is not open, camera_address={stream_address},"
                f" Sleep 5 second.")
            time.sleep(5)
            try:
                camera = cv2.VideoCapture(stream_address)
            except Exception as ex:
                LOGGER.error(f'Unable to access stream [{ex}]')
            continue

        if nframe % fps:
            nframe += 1
            continue

        if nowTime - startTime > 1/fps:
            ret, input_yuv = camera.read()
            startTime = time.time() # reset time

            img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_BGR2RGB)
            nframe += 1
            LOGGER.debug(f"Camera is open, current frame index is {nframe}")
            threading.Thread(target=edge_worker.inference, args=(img_rgb,), daemon=False).start()
            #edge_worker.inference(img_rgb)

if __name__ == '__main__':
    result = retrieve_rtsp_stream()
    start_stream_acquisition(result)
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

import datetime
from distutils import util
import os
import time
import cv2
from urllib.request import Request, urlopen
from estimator import str_to_estimator_class
from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking.utils import get_parameters
from sedna.datasources.obs.connector import OBSClientWrapper
from sedna.core.multi_edge_tracking.components.detector import ObjectDetector

class Bootstrapper():
    def __init__(self) -> None:
        LOGGER.info("Creating Detection/Tracking Bootstrapper module")

        self.estimator_class = get_parameters('estimator_class', "ByteTracker")
        self.hostname = get_parameters('hostname', "unknown")
        self.fps = float(get_parameters('fps', 25))
        self.batch_size = int(get_parameters('batch_size', 1))
        self.video_id = get_parameters('video_id', 0)
        self.video_address = get_parameters('video_address', "") 

        self.eclass = str_to_estimator_class(estimator_class=self.estimator_class)

        self.enable_obs =  bool(util.strtobool(get_parameters('ENABLE_OBS', "False")))
        
        if self.enable_obs:
            self.obs_client = OBSClientWrapper(app_token=get_parameters('OBS_TOKEN', ''))
        
        self.service = None
        
    def run(self):
        protocol = self.video_address.split(":")[0]
        LOGGER.info(f"Detected video source protocol {protocol} for video source {self.video_address}.")

        # TODO: This is not reliable. For example, HLS won't work (https://en.wikipedia.org/wiki/HTTP_Live_Streaming).
        if protocol in ["rtmp", "rtsp"]: #stream
            self.process_video_from_stream()
        elif protocol in ["http"]: #cdn
            filename = self.download_video(protocol)
            self.process_video_from_disk(filename)
        elif os.path.isfile(self.video_address): #file from disk (preloaded)
            self.process_video_from_disk(self.video_address)
        else: # file from obs?
            filename = self.obs_client.download_single_object(self.video_address, ".")
            if filename:
                self.process_video_from_disk(filename)
            else:
                LOGGER.error(f"Unable to open {self.video_address}.")

        self.close()

    def download_video(self):
        try:
            req = Request(self.video_address, headers={'User-Agent': 'Mozilla/5.0'})
            LOGGER.info("Video download complete")
            
            filename = f'{self.video_id}.mp4'
            with open(filename,'wb') as f:
                f.write(urlopen(req).read())
            
            return filename

        except Exception as ex:
            LOGGER.error(f"Unable to download video file {ex}")

    def connect_to_camera(self, stream_address):
        camera = None
        while camera == None or not camera.isOpened():
            try:
                camera = cv2.VideoCapture(stream_address)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            except Exception as ex:
                LOGGER.error(f'Unable to open video source: [{ex}]')
            time.sleep(1)

        return camera

    def process_video_from_disk(self, filename, timeout=20):
        selected_estimator=self.eclass(video_id=self.video_id)
        self.service = ObjectDetector(models=[selected_estimator])

        cap = cv2.VideoCapture(filename)
        index = 0

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if ret == True:
                LOGGER.debug(f"Current frame index is {index}.")
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det_time = datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
                self.service.put((img_rgb, det_time, index))
                index +=1
            else: 
                break

        # When everything done, release the video capture object
        cap.release()

    def process_video_from_stream(self, timeout=20):
        selected_estimator=self.eclass(video_id=self.video_id)
        self.service=ObjectDetector(models=[selected_estimator], asynchronous=True)

        nframe = 0
        grabbed = False
        last_snapshot = time.time()

        camera = self.connect_to_camera(self.video_address)

        while (camera.isOpened()):           
            grabbed = camera.grab()
            
            if grabbed:
                if ((time.time() - last_snapshot) >= 1/self.fps):
                    LOGGER.debug(f"Current frame index is {nframe}.")
                    ret, frame = camera.retrieve()

                    if ret:
                        cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                        det_time = datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
                        self.service.put(data=(frame, det_time, nframe))
                        last_snapshot = time.time()
                nframe += 1

            elif (time.time() - last_snapshot) >= timeout:
                LOGGER.debug(f"Timeout reached, releasing video source.")
                camera.release()
    
    def close(self, timeout=20):
        while (time.time() - self.service.heartbeat) <= timeout:
            LOGGER.debug(f"Waiting for more data from the feature extraction service..")
            time.sleep(1)

        #perform cleanup of the service
        self.service.close()
        LOGGER.info(f"VideoAnalysis job completed.")

if __name__ == '__main__':
    bs = Bootstrapper()
    bs.run()
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

import pickle
import io
from datetime import datetime
import json, base64
import queue
import cv2
from PIL import Image

from sedna.core.multi_edge_tracking import ReIDManagerService
from sedna.common.log import LOGGER

from components.rtsp_dispatcher import add_rtsp_stream, get_rtsp_stream

class Interface():

    def __init__(self, **kwargs):
        LOGGER.info("Starting API interface module")

        # Service settings
        self.target_image = None
        self.target_image_format = "JPEG"
        self.last_update = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
        self.op_mode = "detection"

        # Frame post-processing parameters
        self.post_process = True
        self.encoding = "jpeg"
        self.scaling_factor = 0.5

        # ReID frame buffer
        self.frame_buffer = queue.Queue() #FIFO

    def upload_frame(self, frame):
        LOGGER.info("Received reid result")
        try:
            time = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
            self.frame_buffer.put(frame) if not self.post_process else self.frame_buffer.put((self._post_process(frame, time))) 
            return 200
        except Exception as ex:
            LOGGER.error(f"Unable to add new frame to in-memory buffer. [{ex}]")
            return None

    def get_reid_buffer_size(self):
        LOGGER.info("Return frame buffer size")
        try:
            return self.frame_buffer.qsize()
        except Exception as ex:
            LOGGER.error(f"Unable to get frame buffer size. [{ex}]")
            return 0

    def add_video_address(self, url):
        LOGGER.info("Adding new RTSP stream to list")
        try:
            return add_rtsp_stream(url)
        except Exception as ex:
            LOGGER.error(f"Error while adding RTSP stream. [{ex}]")
            return None

    def get_video_address(self):
        LOGGER.info("Extracting RTSP stream from available ones")
        try:
            return get_rtsp_stream()
        except Exception as ex:
            LOGGER.error(f"Error while retrieving RTSP stream address. [{ex}]")
            return None

    def set_app_details(self, op_mode, target=None):
        LOGGER.info("Updating service configuration")
        try:
            self.op_mode = op_mode

            if target:

                img_bytes = target
                self.target_image = Image.open(io.BytesIO(img_bytes))

                # the image format has to be obtained from here because the open() is lazy
                with Image.open(io.BytesIO(img_bytes)) as im:
                    self.target_image_format = im.format
            
            self.last_update = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")

            return 100

        except Exception as ex:
            LOGGER.error(f"Unable to update service settings. [{ex}]")
            return None

    def get_app_details(self, body):
        LOGGER.info("Retrieving service configuration")
        try:
            response = {}

            try:
                last_update = body.decode('utf-8')
            except Exception as ex:
                LOGGER.error(f"Unable to retrieve last update timestamp. Forcing update. [{ex}]")
                last_update = ""
            
            if self.last_update != last_update:
                LOGGER.info("Pods configuration is obsolete! Sending update now!")

                response = {
                    "op_mode" : self.op_mode,
                    "last_update": self.last_update,
                    "target" :  base64.b64encode(self._image_to_byte_array(self.target_image)).decode("utf-8") if self.target_image else ""
                }

        except Exception as ex:
            LOGGER.error(f"Unable to update pods configuration. [{ex}]")
        
        return response

    def get_reid_result(self):
        LOGGER.info("Fetching latest result")
        try:
            return self.frame_buffer.get(block=False)
        except queue.Empty as ex:
            return None

    def _image_to_byte_array(self, image : Image):
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, self.target_image_format)
        imgByteArr = imgByteArr.getvalue()
        return imgByteArr

    def _write_text_on_image(self, img, text, textX, textY):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        # textX = (img.shape[1] - textsize[0]) / 2
        # textY = (img.shape[0] + textsize[1]) / 2

        # add text centered on image
        cv2.putText(img, text, (textX, textY), font, 1, (0, 255, 0), 2)
    
    def _post_process(self, data, time):
        data = pickle.loads(data)
        try:
            image = cv2.imdecode(data.scene, cv2.IMREAD_COLOR)

            for idx, bbox in enumerate(data.bbox_coord):
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255, 0), 2)
            
                # Add ID
                self._write_text_on_image(image, data.ID[0], int(bbox[0]), int(bbox[1]-10))
            
            # Add camera
            self._write_text_on_image(image, f"Camera:{data.camera[idx]}", 0, 30)

            # Add acquisition delay
            difference = datetime.strptime(time, "%a, %d %B %Y %H:%M:%S.%f") - datetime.strptime(data.detection_time[0], "%a, %d %B %Y %H:%M:%S.%f")
            self._write_text_on_image(image, f"T-Delta:{difference.microseconds/1000}ms", 0, 60)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (0,0), fx = self.scaling_factor, fy = self.scaling_factor)

            # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            output = cv2.imencode("."+ self.encoding, image)[1]
            return output

        except Exception as ex:
            LOGGER.error(f"Error during output scene preparation. {[ex]}")
            return None

# Starting the external API module
reid_manager = ReIDManagerService(interface=Interface)
reid_manager.start()

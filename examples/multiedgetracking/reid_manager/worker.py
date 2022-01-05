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
import threading
from typing import List
import uuid
import io
import time
from datetime import datetime
import base64
import cv2
from PIL import Image
from collections import deque

from sedna.core.multi_edge_tracking.data_classes import DetTrackResult
from sedna.core.multi_edge_tracking import ReIDManagerService
from sedna.common.log import LOGGER
from sedna.common.config import Context

from components.rtsp_dispatcher import add_rtsp_stream, get_rtsp_stream, reset_rtsp_stream_list
from components.rabbitmq import RabbitMQWriter

class ReIDBuffer():
    def __init__(self) -> None:
        self.uuid = str(uuid.uuid4())
        self.frame_buffer = deque()

class Interface():

    def __init__(self, **kwargs):
        LOGGER.info("Starting API interface module")

        # Service settings
        self.target_image = []
        self.target_image_format = "JPEG"
        self.last_update = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
        self.op_mode = "detection"
        self.userid = "DEFAULT"

        # Frame post-processing parameters
        self.post_process = True
        self.encoding = "jpeg"
        self.scaling_factor = 0.5

        # ReID Hashtable (WIP)
        self.reid_ht = {}

        # ReID frame buffer
        self.frame_buffer = deque() #FIFO
        self.rtmp_url="rtmp://7.182.9.110:1935/live/" + self.userid
        # self.rtmp_pipe = self._create_rtmp_pipe()
        threading.Thread(target=self._generate_video, daemon=False).start()

        # Create RAbbitMQ writer
        self.rabbitmq_address = Context.get_parameters('rabbitmq_address', "7.182.9.110")
        self.rabbitmq_port = int(Context.get_parameters('rabbitmq_port', 32672))
        self.rabbitmq_queue = Context.get_parameters('rabbitmq_queue', "reid")

        self.rabbitmq_interface = RabbitMQWriter(address=self.rabbitmq_address, port=self.rabbitmq_port, queue=self.rabbitmq_queue)

    def upload_frame(self, data):
        LOGGER.info("Received reid result")
        try:
            dt_object = pickle.loads(data)
            time = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
            self.frame_buffer.append(dt_object) if not self.post_process else self.frame_buffer.append((self._post_process(dt_object, time)))

            # Write to RabbitMQ
            self.rabbitmq_interface.target_found(self.rtmp_url, dt_object, len(self.frame_buffer) - 1)
            
            # Add frame to RTMP video
            # threading.Thread(target=self._generate_video, args=(dt_object.scene,), daemon=False).start()
            # self._generate_video(dt_object.scene)

            return 200
        except Exception as ex:
            LOGGER.error(f"Unable to add new frame to in-memory buffer. [{ex}]")
            return None

    def get_reid_buffer_size(self):
        LOGGER.info("Return frame buffer size")
        try:
            return len(self.frame_buffer)
        except Exception as ex:
            LOGGER.error(f"Unable to get frame buffer size. [{ex}]")
            return 0

    def add_video_address(self, url, camid=0):
        LOGGER.info("Adding new RTSP stream to list")
        try:
            return add_rtsp_stream(url, camid)
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
    
    def reset_rtsp_stream_list(self):
        LOGGER.info("Clearing RTSP stream list")
        try:
            return reset_rtsp_stream_list()
        except Exception as ex:
            LOGGER.error(f"Error while retrieving RTSP stream address. [{ex}]")
            return None

    def set_app_details(self, op_mode, target=[], userid="123"):
        LOGGER.info("Updating service configuration")
        try:
            self.op_mode = op_mode
            self.userid = userid

            LOGGER.info(f"Received {len(target)} images for the target.")
            k = []
            for image in target:

                img_bytes = image
                k.append(Image.open(io.BytesIO(img_bytes)))

                # the image format has to be obtained from here because the open() is lazy
                with Image.open(io.BytesIO(img_bytes)) as im:
                    self.target_image_format = im.format
            
            self.last_update = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
            self.target_image = k

            # self.reid_ht[userid] = ReIDBuffer()

            return 100

        except Exception as ex:
            LOGGER.error(f"Unable to update service settings. [{ex}]")
            return None

    def set_app_details_v2(self, userid="123", op_mode="detection", target=[]):
        LOGGER.info("Updating service configuration")
        try:
            self.op_mode = op_mode
            self.userid = userid

            LOGGER.info(f"Received {len(target)} images for the target.")
            k = []
            for image in target:

                img_bytes = base64.b64decode(image.encode('utf-8'))
                k.append(Image.open(io.BytesIO(img_bytes)))

                # the image format has to be obtained from here because the open() is lazy
                with Image.open(io.BytesIO(img_bytes)) as im:
                    self.target_image_format = im.format
            
            self.last_update = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
            self.target_image = k

            return 100

        except Exception as ex:
            LOGGER.error(f"Unable to update service settings. [{ex}]")
            return None


    def get_app_details(self, body):
        LOGGER.debug("Retrieving service configuration")
        try:
            response = {}

            try:
                last_update = body.decode('utf-8')
            except Exception as ex:
                LOGGER.error(f"Unable to retrieve last update timestamp. Forcing update. [{ex}]")
                last_update = ""
            
            if self.last_update != last_update:
                LOGGER.info("Pods configuration is obsolete! Sending update now!")

                
                target_images = []
                for image in self.target_image:
                    target_images.append(base64.b64encode(self._image_to_byte_array(image)).decode("utf-8"))

                response = {
                    "op_mode" : self.op_mode,
                    "last_update": self.last_update,
                    "target" :  target_images if len(target_images) > 0 else ""
                }

        except Exception as ex:
            LOGGER.error(f"Unable to update pods configuration. [{ex}]")
        
        return response

    def get_reid_result(self):
        LOGGER.info("Fetching latest result")
        try:
            dt_object : DetTrackResult = self.frame_buffer.popleft()
            return dt_object.scene
        except IndexError as ex:
            return None

    def clean_frame_buffer(self):
        LOGGER.info("Clean frame buffer")
        self.frame_buffer.clear()

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
    
    # Post-process will overwrite the original scene with a new version containing ReID information
    def _post_process(self, data, time):
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

            output = cv2.imencode("."+ self.encoding, image)[1]
            
            data.scene = output

            return data

        except Exception as ex:
            LOGGER.error(f"Error during output scene preparation. {[ex]}")
            return None

    # Called only once to create the pipe to write to FFMPEG
    def _create_rtmp_pipe(self):
        import subprocess as sp
        LOGGER.info("Create RTMP pipe")

        fps = 1
        width = 640
        height = 480
        
        command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f"{width}x{height}",		# weight and height of your image
                '-r', str(fps),					# fps defined,
                '-i', '-',
                '-r', '1',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                '-flvflags', 'no_duration_filesize',
                self.rtmp_url]
                
        pipe_push = sp.Popen(command, stdin=sp.PIPE, shell=False)

        return pipe_push

    # This is called every n seconds to generate the updated version of the video stream
    def _generate_video(self, regeneration_interval=2, expansion_factor=100):
        prev_len = 0

        while True:
            # We create a copy of the frame buffer
            lb = self.frame_buffer.copy()

            if len(lb) > prev_len: 
                LOGGER.info("Reconstructing ReID video")
                pipe = self._create_rtmp_pipe()
                try:                   
                    for item in lb:
                        image = cv2.imdecode(item.scene, cv2.IMREAD_COLOR)
                        image = cv2.resize(image, (640,480))

                        # As the number of frames at our disposal is small,
                        # we replicate the same frame N times. We do so to 
                        # have enough frames in the output buffer for FFMPEG
                        # to create a renderable stream.
                        for i in range(expansion_factor):
                            pipe.stdin.write(image)
                    # pipe.stdin.flush() # has no effect
                    # out, err = pipe.communicate() # cannot be used because it kills the pipe
                    # LOGGER.info(out)
                    # LOGGER.info(err)
                except Exception as ex:
                    LOGGER.error(f"Error during transmission to RTMP server. [{ex}]")

                pipe.stdin.close()
                prev_len = len(lb)

            time.sleep(regeneration_interval)
    


# Starting the external API module
reid_manager = ReIDManagerService(interface=Interface)
reid_manager.start()

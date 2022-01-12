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
from typing import Dict
import uuid
import io
import time
from datetime import datetime
import base64
import cv2
from PIL import Image
from collections import deque
from sedna.common.benchmark import FTimer
from sedna.core.multi_edge_tracking.data_classes import SyncDS, DetTrackResult, TargetImages, OP_MODE

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

        # Service parameters
        self.sync_object = SyncDS()

        # Frame post-processing parameters
        self.post_process = True
        self.encoding = "jpeg"
        self.scaling_factor = 0.5

        # ReID Dict (WIP)
        self.reid_ht : Dict[str, ReIDBuffer] = dict() 

        # RTMP parameters
        self.rtmp_url="rtmp://7.182.9.110:1935/live/"
        threading.Thread(target=self._generate_video, daemon=False).start()

        # RabbitMQ parameters
        self.rabbitmq_address = Context.get_parameters('rabbitmq_address', "7.182.9.110")
        self.rabbitmq_port = int(Context.get_parameters('rabbitmq_port', 32672))
        self.rabbitmq_queue = Context.get_parameters('rabbitmq_queue', "reid")

        self.rabbitmq_interface = RabbitMQWriter(address=self.rabbitmq_address, port=self.rabbitmq_port, queue=self.rabbitmq_queue)

    def upload_frame(self, data):
        LOGGER.info("Store ReID result")
        time = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")

        try:
            dt_object_list = pickle.loads(data)
        except Exception as ex:
            LOGGER.error(f"Unable to access received data. [{ex}]")
            return None

        # As we receive a list of objects with results for each userid, we iterate over it. 
        for dt_object in dt_object_list:
            try:
                user_buffer = self.reid_ht[dt_object.userID].frame_buffer
            except KeyError:
                # If the buffer doesn't exists, we create it.
                # This helps in case of ReID Manager restart/crash
                # to avoid information loss.
                self.reid_ht[dt_object.userID] = ReIDBuffer()
                user_buffer = self.reid_ht[dt_object.userID].frame_buffer

            seq_number = len(user_buffer)
            user_buffer.append(dt_object) if not self.post_process else user_buffer.append((self._post_process(dt_object, time, seq_number)))

            # Write to RabbitMQ
            self.rabbitmq_interface.target_found(self.rtmp_url + dt_object.userID, dt_object.userID, dt_object, seq_number - 1)
            
        return 200


    def get_reid_buffer_size(self, userid="DEFAULT"):
        LOGGER.info("Return frame buffer size")
        try:
            return len(self.reid_ht[userid].frame_buffer)
        except Exception as ex:
            LOGGER.error(f"Unable to get frame buffer size. [{ex}]")
            return 0

    def add_video_address(self, url, camid=0, receiver="unknown"):
        LOGGER.info("Add new RTSP stream to list")
        try:
            return add_rtsp_stream(url, camid, receiver)
        except Exception as ex:
            LOGGER.error(f"Error while adding RTSP stream. [{ex}]")
            return None

    def get_video_address(self, hostname):
        LOGGER.info("Extract RTSP stream from available ones")
        try:
            return get_rtsp_stream(hostname)
        except Exception as ex:
            LOGGER.error(f"Error while retrieving RTSP stream address. [{ex}]")
            return None
    
    def reset_rtsp_stream_list(self):
        LOGGER.info("Clear RTSP stream list")
        try:
            return reset_rtsp_stream_list()
        except Exception as ex:
            LOGGER.error(f"Error while retrieving RTSP stream address. [{ex}]")
            return None

    def set_app_details(self, op_mode, target=[], userid="DEFAULT"):
        LOGGER.info("Update service configuration")
        try:
            self.sync_object.op_mode = OP_MODE(op_mode)

            LOGGER.info(f"Received {len(target)} images for the target for user {userid}.")
            temp = []
            for image in target:
                temp.append(Image.open(io.BytesIO(image)))

                # the image format has to be obtained from here because the open() is lazy
                # with Image.open(io.BytesIO(img_bytes)) as im:
                #     self.target_image_format = im.format
            
            self.sync_object.last_update = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
            self._update_targets_collection(userid, temp)

            self.reid_ht[userid] = ReIDBuffer()

            return 200

        except Exception as ex:
            LOGGER.error(f"Unable to update service settings. [{ex}]")
            return None

    # TODO: Merge with set_app_details
    def set_app_details_v2(self, userid="DEFAULT", op_mode="detection", target=[]):
        LOGGER.info("Update service configuration")
        try:
            self.sync_object.op_mode = OP_MODE(op_mode)

            LOGGER.info(f"Received {len(target)} images for the target.")
            temp = []
            for image in target:

                img_bytes = base64.b64decode(image.encode('utf-8'))
                temp.append(Image.open(io.BytesIO(img_bytes)))

                # the image format has to be obtained from here because the open() is lazy
                # with Image.open(io.BytesIO(img_bytes)) as im:
                #     self.target_image_format = im.format
            
            self.sync_object.last_update = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
            self._update_targets_collection(userid, temp)

            self.reid_ht[userid] = ReIDBuffer()

            return 200

        except Exception as ex:
            LOGGER.error(f"Unable to update service settings. [{ex}]")
            return None


    def get_app_details(self, body):
        LOGGER.debug("Retrieve service configuration")
        try:
            response = None

            try:
                last_update = body.decode('utf-8')
            except Exception as ex:
                LOGGER.error(f"Unable to retrieve last update timestamp. Forcing update. [{ex}]")
                last_update = ""
            
            if self.sync_object.last_update != last_update:
                LOGGER.info("Pods configuration is obsolete! Sending update now!")

                # target_images = []
                # for image in self.target_image:
                #     target_images.append(base64.b64encode(self._image_to_byte_array(image)).decode("utf-8"))

                # response = {
                #     "op_mode" : self.op_mode,
                #     "last_update": self.last_update,
                #     "target" :  target_images if len(target_images) > 0 else ""
                # }

                #TODO: Implement delta updates?
                response = pickle.dumps(self.sync_object)

        except Exception as ex:
            LOGGER.error(f"Unable to update pods configuration. [{ex}]")
        
        return response

    def get_reid_result(self, userid="DEFAULT"):
        LOGGER.info(f"Fetch frames from user {userid} buffer.")
        try:
            dt_object : DetTrackResult = self.reid_ht[userid].frame_buffer.popleft()
            return dt_object.scene
        except IndexError as ex:
            return None

    def clean_frame_buffer(self, userid="DEFAULT"):
        LOGGER.info(f"Clear frame buffer for user {userid}")
        try:
            self.reid_ht[userid].frame_buffer.clear()
        except Exception as ex:
            LOGGER.error(f"Error while cleaning ReID buffer for user {userid}. [{ex}]")
   
    def _update_targets_collection(self, userid, targets):
        target = list(filter(lambda x: x.userid == userid, self.sync_object.targets_collection))

        if len(target) > 0:
            LOGGER.info(f"Update existing features for user {userid}")
            target[0].targets = targets
        else:
            LOGGER.info(f"Create new list of features for user {userid}")
            self.sync_object.targets_collection.append(TargetImages(userid, targets))

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
    def _post_process(self, data, time, seq_number):
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

            # Add sequence number (relative to the local ReID buffer)
            self._write_text_on_image(image, f"Seq#:{seq_number}", 0, 90)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (0,0), fx = self.scaling_factor, fy = self.scaling_factor)

            output = cv2.imencode("."+ self.encoding, image)[1]
            
            data.scene = output

            return data

        except Exception as ex:
            LOGGER.error(f"Error during output scene preparation. {[ex]}")
            return None

    # Called only once to create the pipe to write to FFMPEG
    # TODO: Use python-ffmpeg library rather than subprocess..
    def _create_rtmp_pipe(self, userid):
        import subprocess as sp
        LOGGER.debug("Create RTMP pipe")

        fps = 1
        width = 640
        height = 480
        
        command = ['ffmpeg',
                '-y',
                '-stream_loop', '-1',
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
                self.rtmp_url+userid]

        # command = ['ffmpeg',
        #             '-re',
        #             '-i', f'vittorio.mkv',
        #             '-f', 'flv',
        #             self.rtmp_url+userid
        #             ]
                
        pipe_push = sp.Popen(command, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.DEVNULL, shell=False)

        return pipe_push

    def _create_video(self):
        for buffer in self.reid_ht.items():
            lb = buffer[1].frame_buffer.copy()
            out = cv2.VideoWriter(f'{buffer[0]}.mkv',cv2.VideoWriter_fourcc(*'FLV1'), 1, (640,480))
            for item in lb:
                image = cv2.imdecode(item.scene, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (640,480))
                out.write(image)
            
            out.release()
    
    # This is called every n seconds to generate the updated version of the video stream
    # TODO: This can become a very intensive operation depending on the amount of frames in each
    # buffer. At some point, this needs to be improved.
    def _generate_video(self, regeneration_interval=5, expansion_factor=100):
        prev_len = 0

        while True:
            # As there could be many buffers, we create a video stream separately for each one.
            for buffer in self.reid_ht.items():
                # We create a copy of the frame buffer
                lb = buffer[1].frame_buffer.copy()

                if len(lb) != prev_len: 
                    LOGGER.info(f"Reconstruct ReID video for user {buffer[0]}")

                    with FTimer("video_reconstruction"):
                        pipe = self._create_rtmp_pipe(buffer[0])

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
                        
            # self._create_video()
            time.sleep(regeneration_interval)

    def _generate_video_test(self, regeneration_interval=5, expansion_factor=100):
        prev_len = 0

        while True:
            self._create_video()
            self._create_rtmp_pipe("vittorio")

            time.sleep(regeneration_interval)
    


# Starting the external API module
reid_manager = ReIDManagerService(interface=Interface)
reid_manager.start()

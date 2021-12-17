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

import io
import json
import numpy as np

from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse, StreamingResponse, Response

from .base import BaseServer

__all__ = ('ReIDManagerServer', )

class ReIDManagerServer(BaseServer):  # pylint: disable=too-many-arguments
    """
    external api server
    """

    def __init__(
            self,
            interface,
            servername,
            host: str = '127.0.0.1',
            http_port: int = 8080,
            max_buffer_size: int = 104857600,
            workers: int = 1):
        super(
            ReIDManagerServer,
            self).__init__(
            servername=servername,
            host=host,
            http_port=http_port,
            workers=workers)
        self.interface=interface
        self.max_buffer_size = max_buffer_size
        self.app = FastAPI(
            routes=[
                APIRoute(
                    f"/{servername}/get_reid_result",
                    self.get_reid_result,
                    methods=["GET"],
                ),
                APIRoute(
                    f"/{servername}/upload_data",
                    self.upload_frame,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{servername}/get_video_address",
                    self.get_video_address,
                    response_class=JSONResponse,
                    methods=["GET"],
                ),
                APIRoute(
                    f"/{servername}/add_video_address",
                    self.add_video_address,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{servername}/reset_rtsp_stream_list",
                    self.reset_rtsp_stream_list,
                    response_class=JSONResponse,
                    methods=["GET"],
                ),
                APIRoute(
                    f"/{servername}/get_app_details",
                    self.get_app_details,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{servername}/set_app_details",
                    self.set_app_details,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{servername}/get_reid_buffer_size",
                    self.get_reid_buffer_size,
                    response_class=JSONResponse,
                    methods=["GET"],
                ),
                APIRoute(
                    f"/{servername}/enable_post_processing",
                    self.enable_post_processing,
                    response_class=JSONResponse,
                    status_code=200,
                    methods=["GET"],
                ),
                APIRoute(
                    f"/{servername}/disable_post_processing",
                    self.disable_post_processing,
                    response_class=JSONResponse,
                    methods=["GET"],
                ),
                APIRoute(
                    f"/{servername}/clean_frame_buffer",
                    self.clean_frame_buffer,
                    response_class=JSONResponse,
                    methods=["GET"],
                ),
                # EXTRA API FOR CT#
                APIRoute(
                    f"/v1/person/user/tracking/stop",
                    self.stop_tracking,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/v1/person/tracking/live/identification",
                    self.set_app_details_v2,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
            ],
            log_level="trace",
            timeout=600,
        )

    def start(self):
        return self.run(self.app)

    ##########################
    ### EXTERNAL ENDPOINTS ###
    ##########################

    # Example: curl -X GET http://7.182.9.110:9907/sedna/enable_post_processing
    # Enables post-processing which makes the service store images ready for rendering rather than the parent DetTrackResult object
    def enable_post_processing(self):
        self.interface.post_process = True
        return 200

    # Example: curl -X GET http://7.182.9.110:9907/sedna/disable_post_processing
    # Disables post-processing.
    def disable_post_processing(self):
        self.interface.post_process = False
        return 200

    # Example: curl -X GET http://7.182.9.110:9907/sedna/clean_frame_buffer
    # Wipes the frame buffer
    def clean_frame_buffer(self):
        self.interface.clean_frame_buffer()

    # Example: curl -X GET http://7.182.9.110:9907/sedna/get_reid_result
    # Returns the oldest from the frame buffer (FIFO queue)
    async def get_reid_result(self):
        data = self.interface.get_reid_result()

        # If this is True, it means that we are sending out an image
        if isinstance(data, (np.ndarray, np.generic) ):
            return StreamingResponse(io.BytesIO(data.tobytes()), media_type="image/"+ self.interface.encoding)
        
        # Otherwise, we are sending a buffer with a DetTrackResult object
        if isinstance(data, (bytearray, bytes)):
             return StreamingResponse(content=data)

        return Response(content="NO FRAMES!")

    # Example: curl -X GET http://7.182.9.110:9907/sedna/get_reid_buffer_size
    # Returns the size of the frame buffer
    def get_reid_buffer_size(self):
        return self.interface.get_reid_buffer_size()

    # Example: curl -X POST http://7.182.9.110:9907/sedna/add_video_address --data '{"url":"rtsp://localhost:8080/video/0", "camid":0}'
    # Add a new RTSP address to the list
    async def add_video_address(self, request: Request):
        body = await request.body()
        body = json.loads(body)

        url = body.get('url', None) 
        camid = body.get('camid', None)

        return self.interface.add_video_address(url, camid)


    # Example: curl -X GET http://7.182.9.110:9907/sedna/reset_rtsp_stream_list
    # Reset RTSP addresses list
    async def reset_rtsp_stream_list(self, request: Request):
        return self.interface.reset_rtsp_stream_list()

    # Example: curl -X POST http://7.182.9.110:9907/sedna/set_app_details  -H 'Expect:' -F data='{"userID":"123", "op_mode":"tracking", "queryImagesFromNative": []}' -F target=@vit_vid.png  target=@zi_vid.png
    # Updates the service configuration. It accepts a string specifing the operation mode and a file containing the target to search.
    # All images MUST have the same extension (JPG, PNG ..).
    async def set_app_details(self, request: Request):
        form = await request.form()
        data_json = json.loads(form.get("data", "{}"))

        op_mode = data_json.get("op_mode", "detection")
        userID = data_json.get("userID", "123")
        files = form.getlist("target")
   
        target = []
        if files is not None and len(files) > 0:
            for file in files:
                target.append(await file.read()) 

        return self.interface.set_app_details(op_mode, target)

    # Example: curl -X POST http://7.182.9.110:9907/v1/person/tracking/live/identification  -H 'Expect:' --data '{"userID": "123", "op_mode":"tracking", "queryImagesFromNative": [], "cameraIds": [], "isEnhanced": 0}'
    # Updates the service configuration. It accepts a string specifing the operation mode and a file containing the target to search.
    # All images MUST have the same extension (JPG, PNG ..).
    async def set_app_details_v2(self, request: Request):
        body = await request.body()
        data_json = json.loads(body)

        userID = data_json.get("userID", "123")
        op_mode = data_json.get("op_mode", "detection")
        isEnhanced = data_json.get("isEnhanced", 0) #not used
        queryImagesFromNative = data_json.get("queryImagesFromNative", [])
        cameraIds = data_json.get("cameraIds", []) #not used

        return self.interface.set_app_details_v2(userID, op_mode, queryImagesFromNative)

    # Example: curl -X GET http://7.182.9.110:9907/sedna/get_reid_buffer_size
    # Returns the size of the frame buffer
    def stop_tracking(self):
        """Not Implemented """
        return 200

    ##########################
    ### INTERNAL ENDPOINTS ###
    ##########################

    def get_video_address(self):
        return self.interface.get_video_address()

    async def upload_frame(self, request: Request):
        body = await request.body()
        return self.interface.upload_frame(body)

    async def get_app_details(self, request: Request):
        body = await request.body()
        return self.interface.get_app_details(body)

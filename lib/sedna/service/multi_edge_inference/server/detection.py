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

from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse

from sedna.service.server.base import BaseServer

__all__ = ('DetectionServer', )


class DetectionServer(BaseServer):  # pylint: disable=too-many-arguments
    """
    REST api server for object detection component
    """

    def __init__(
            self,
            model,
            service_name,
            ip: str = '127.0.0.1',
            port: int = 8080,
            max_buffer_size: int = 1004857600,
            workers: int = 1):
        super(
            DetectionServer,
            self).__init__(
            servername=service_name,
            host=ip,
            http_port=port,
            workers=workers)

        self.model = model
        self.max_buffer_size = max_buffer_size
        self.app = FastAPI(
            routes=[
                APIRoute(
                    f"/{service_name}/update_service",
                    self.update_service,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{service_name}/video_analytics",
                    self.video_analytics,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{service_name}/status",
                    self.status,
                    response_class=JSONResponse,
                    methods=["GET"],
                ),
            ],
            log_level="trace",
            timeout=600,
        )

    def start(self):
        return self.run(self.app)

    def status(self, request: Request):
        return "OK"

    async def video_analytics(self, request: Request):
        s = await request.body()
        self.model.put([pickle.loads(s)])

        return 200

    async def update_service(self, request: Request):
        s = await request.body()
        return self.model.update_operational_mode(pickle.loads(s))

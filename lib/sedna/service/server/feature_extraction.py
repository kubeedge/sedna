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
from typing import List, Optional

from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse

from .base import BaseServer

__all__ = ('FEServer', )


class ServeModelInfoResult(
        BaseModel):  # pylint: disable=too-few-public-methods
    """
    Expose model information
    """

    infos: List


class ServePredictResult(BaseModel):  # pylint: disable=too-few-public-methods
    """
    result
    """

    result: List


class InferenceItem(BaseModel):  # pylint: disable=too-few-public-methods
    data: bytes
    callback: Optional[str] = None


class FEServer(BaseServer):  # pylint: disable=too-many-arguments
    """
    rest api server for reid
    """

    def __init__(
            self,
            model,
            servername,
            host: str = '127.0.0.1',
            http_port: int = 8080,
            max_buffer_size: int = 104857600,
            workers: int = 1):
        super(
            FEServer,
            self).__init__(
            servername=servername,
            host=host,
            http_port=http_port,
            workers=workers)
        self.model = model
        self.max_buffer_size = max_buffer_size
        self.app = FastAPI(
            routes=[
                APIRoute(
                    f"/{servername}",
                    self.model_info,
                    response_model=ServeModelInfoResult,
                    response_class=JSONResponse,
                    methods=["GET"],
                ),
                APIRoute(
                    f"/{servername}/feature_extraction",
                    self.feature_extraction,
                    response_model=ServePredictResult,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
            ],
            log_level="trace",
            timeout=600,
        )

    def start(self):
        return self.run(self.app)

    def model_info(self):
        return ServeModelInfoResult(infos=self.get_all_urls())

    async def feature_extraction(self, request: Request):
        s = await request.body()
        self.model.put_data([pickle.loads(s)])
        # self.model.put_data(data.data[0].from_json())
        # inference_res = self.model.inference(data.data, post_process=data.callback)
        
        return ServePredictResult(result=[])

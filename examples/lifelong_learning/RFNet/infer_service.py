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

import os
import threading
import time
from datetime import datetime
from io import BytesIO
from typing import Optional, Any

import cv2
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import uvicorn
from torchvision import transforms
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sedna.common.utils import get_host_ip

from sedna_predict import init_ll_job, preprocess
from dataloaders.datasets.cityscapes import CityscapesSegmentation
from basemodel import Model
from dataloaders import custom_transforms as tr

class ImagePayload(BaseModel):
    image: UploadFile = File(...)
    depth: Optional[UploadFile] = None


class ResultModel(BaseModel):
    type: int = 0
    box: Any = None
    curr: str = None
    future: str = None
    img: str = None


class ResultResponse(BaseModel):
    msg: str = ""
    result: Optional[ResultModel] = None
    code: int


class BaseServer:
    # pylint: disable=too-many-instance-attributes,too-many-arguments
    DEBUG = True
    WAIT_TIME = 15

    def __init__(
            self,
            servername: str,
            host: str,
            http_port: int = 8080,
            grpc_port: int = 8081,
            workers: int = 1,
            ws_size: int = 16 * 1024 * 1024,
            ssl_key=None,
            ssl_cert=None,
            timeout=300):
        self.server_name = servername
        self.app = None
        self.host = host or '0.0.0.0'
        self.http_port = http_port or 80
        self.grpc_port = grpc_port
        self.workers = workers
        self.keyfile = ssl_key
        self.certfile = ssl_cert
        self.ws_size = int(ws_size)
        self.timeout = int(timeout)
        protocal = "https" if self.certfile else "http"
        self.url = f"{protocal}://{self.host}:{self.http_port}"

    def run(self, app, **kwargs):
        if hasattr(app, "add_middleware"):
            app.add_middleware(
                CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                allow_methods=["*"], allow_headers=["*"],
            )

        uvicorn.run(
            app,
            host=self.host,
            port=self.http_port,
            ssl_keyfile=self.keyfile,
            ssl_certfile=self.certfile,
            workers=self.workers,
            timeout_keep_alive=self.timeout,
            **kwargs)

    def get_all_urls(self):
        url_list = [{"path": route.path, "name": route.name}
                    for route in getattr(self.app, 'routes', [])]
        return url_list


class InferenceServer(BaseServer):  # pylint: disable=too-many-arguments
    """
    rest api server for inference
    """

    def __init__(
            self,
            servername,
            host: str,
            http_port: int = 30001,
            max_buffer_size: int = 104857600,
            workers: int = 1):
        super(
            InferenceServer,
            self).__init__(
            servername=servername,
            host=host,
            http_port=http_port,
            workers=workers)

        self.ll_job = init_ll_job()
        # self.model = Model(num_class=31)
        # self.model.load("./models/2048x1024_80.pth")

        self.max_buffer_size = max_buffer_size
        self.app = FastAPI(
            routes=[
                APIRoute(
                    f"/{servername}",
                    self.model_info,
                    methods=["GET"],
                ),
                APIRoute(
                    f"/{servername}/predict",
                    self.predict,
                    methods=["POST"],
                    response_model=ResultResponse
                ),
            ],
            log_level="trace",
            timeout=600,
        )
        self.index_frame = 0

    def start(self):
        return self.run(self.app)

    @staticmethod
    def model_info():
        return HTMLResponse(
            """<h1>Welcome to the RestNet API!</h1>
            <p>To use this service, send a POST HTTP request to {this-url}/predict</p>
            <p>The JSON payload has the following format: {"image": "BASE64_STRING_OF_IMAGE", 
            "depth": "BASE64_STRING_OF_DEPTH"}</p>
            """)

    async def predict(self, image: UploadFile = File(...), depth: Optional[UploadFile] = None) -> ResultResponse:
        contents = await image.read()
        self.image = Image.open(BytesIO(contents)).convert('RGB')
        self.image.save(f"/home/lsq/curb_images/{str(time.time())}.png")

        self.index_frame = self.index_frame + 1
        
        img_rgb = cv2.resize(np.array(self.image), (2048, 1024), interpolation=cv2.INTER_CUBIC)
        img_rgb = Image.fromarray(img_rgb)
        if depth:
            depth_contents = await depth.read()
            depth = Image.open(BytesIO(depth_contents)).convert('RGB')
            img_dep = cv2.resize(np.array(depth), (2048, 1024), interpolation=cv2.INTER_CUBIC)
            img_dep = Image.fromarray(img_dep)
        else:
            img_dep = img_rgb

        sample = {'image': img_rgb, "depth": img_dep, "label": img_rgb}
        predict_data = preprocess(sample)
        results, _, _ = self.ll_job.inference(predict_data)

        # self.model.validator.test_loader = DataLoader(
        #     predict_data.x,
        #     batch_size=self.model.val_args.test_batch_size,
        #     shuffle=False,
        #     pin_memory=True)
        # results = self.model.validator.validate()

        results = post_process(results)
        curr, future = get_curb(results["result"]["box"], img_rgb)
        results["result"]["curr"] = curr
        results["result"]["future"] = future
        results["result"]["box"] = None

        print(results)
        return results


def parse_result(label, count):
    label_map = ['road', 'sidewalk']
    count_d = dict(zip(label, count))
    curb_count = count_d.get(19, 0)
    if curb_count / np.sum(count) > 0.2: # > 0.3
        return "curb"
    r = sorted(label, key=count_d.get, reverse=True)[0]
    try:
        c = label_map[r]
    except:
        c = "other"

    return c

def get_curb(results, img_rgb):
    results = np.array(results[0])
    input_height, input_width = results.shape
    
    # small trapezoid
    closest = np.array([
        [0, int(input_height)],
        [int(input_width),
         int(input_height)],
        [int(0.882 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(0.118 * input_width + .5),
         int(.8 * input_height + .5)]
    ])

    future = np.array([
        [int(0.118 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(0.882 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(.765 * input_width + .5),
         int(.66 * input_height + .5)],
        [int(.235 * input_width + .5),
         int(.66 * input_height + .5)]
    ])

    # big trapezoid
    # closest = np.array([
    #     [0, int(input_height)],
    #     [int(input_width),
    #      int(input_height)],
    #     [int(0.882 * input_width + .5),
    #      int(.7 * input_height + .5)],
    #     [int(0.118 * input_width + .5),
    #      int(.7 * input_height + .5)]
    # ])
    #
    # future = np.array([
    #     [int(0.118 * input_width + .5),
    #      int(.7 * input_height + .5)],
    #     [int(0.882 * input_width + .5),
    #      int(.7 * input_height + .5)],
    #     [int(.765 * input_width + .5),
    #      int(.5 * input_height + .5)],
    #     [int(.235 * input_width + .5),
    #      int(.5 * input_height + .5)]
    # ])

    # _draw_closest_and_future(closest, future, img_rgb)
    
    mask = np.zeros((input_height, input_width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [closest], 1)
    mask = cv2.fillPoly(mask, [future], 2)
    d1, c1 = np.unique(results[mask == 1], return_counts=True)
    d2, c2 = np.unique(results[mask == 2], return_counts=True)
    c = parse_result(d1, c1)
    f = parse_result(d2, c2)
    
    return c, f

def _draw_closest_and_future(closest, future, img_rgb):
    img_array = np.array(img_rgb)
    closest_color = [0, 191, 255]
    future_color = [255, 69, 0]

    img_array = cv2.fillPoly(img_array, [closest], closest_color)
    img_array = cv2.fillPoly(img_array, [future], future_color)

    new_img = Image.fromarray(img_array)
    new_img.save("test.png")

def post_process(res):
    res = res[0].tolist()
    type = 0
    mesg = {
        "msg": "",
        "result": {
            "type": type,
            "box": res
        },
        "code": 0
    }
    return mesg

if __name__ == '__main__':
    web_app = InferenceServer("lifelong-learning-robo", host=get_host_ip())
    web_app.start()

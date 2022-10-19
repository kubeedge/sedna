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


class ResultResponse(BaseModel):
    msg: str = ""
    result: str = ""
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
            http_port: int = 5000,
            max_buffer_size: int = 104857600,
            workers: int = 1):
        super(
            InferenceServer,
            self).__init__(
            servername=servername,
            host=host,
            http_port=http_port,
            workers=workers)

        self.ll_job = init_ll_job(weight_path="./models/ramp_train1_200.pth")
        # self.model = Model(num_class=31)
        # self.model.load("./models/ramp_train1_200.pth")

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
        self.image.save(f"/home/lsq/ramp_images/{str(time.time())}.png")

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

        msg = {
            "msg": "",
            "result": get_ramp(results[0].tolist(), img_rgb),
            "code": 0
        }
        return msg

def parse_result(label, count, ratio):
    count_d = dict(zip(label, count))
    ramp_count = count_d.get(21, 0)
    if ramp_count / np.sum(count) > ratio:
        return True
    else:
        return False

def get_ramp(results, img_rgb):
    results = np.array(results[0])
    input_height, input_width = results.shape

    # big trapezoid
    big_closest = np.array([
        [0, int(input_height)],
        [int(input_width),
         int(input_height)],
        [int(0.882 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(0.118 * input_width + .5),
         int(.8 * input_height + .5)]
    ])

    big_future = np.array([
        [int(0.118 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(0.882 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(.765 * input_width + .5),
         int(.66 * input_height + .5)],
        [int(.235 * input_width + .5),
         int(.66 * input_height + .5)]
    ])

    # small trapezoid
    small_closest = np.array([
        [488, int(input_height)],
        [1560, int(input_height)],
        [1391, int(.8 * input_height + .5)],
        [621, int(.8 * input_height + .5)]
    ])

    small_future = np.array([
        [741, int(.66 * input_height + .5)],
        [1275, int(.66 * input_height + .5)],
        [1391, int(.8 * input_height + .5)],
        [621, int(.8 * input_height + .5)]
    ])

    upper_left = np.array([
        [1567, 676],
        [1275, 676],
        [1391, 819],
        [1806, 819]
    ])

    bottom_left = np.array([
        [1806, 819],
        [1391, 819],
        [1560, 1024],
        [2048, 1024]
    ])

    upper_right = np.array([
        [741, 676],
        [481, 676],
        [242, 819],
        [621, 819]
    ])

    bottom_right = np.array([
        [621, 819],
        [242, 819],
        [0, 1024],
        [488, 1024]
    ])

    # _draw_closest_and_future((big_closest, big_future), (small_closest, small_future), img_rgb)

    ramp_location = locate_ramp(small_closest, small_future,
                                upper_left, bottom_left,
                                upper_right, bottom_right,
                                results)

    if not ramp_location:
        ramp_location = "no_ramp"

    return ramp_location

def locate_ramp(small_closest, small_future,
                upper_left, bottom_left,
                upper_right, bottom_right,
                results):

    if has_ramp(results, (small_closest, small_future), 0.9, 0.7):
        return "small_trapezoid"

    right_location = has_ramp(results, (bottom_right, upper_right), 0.4, 0.2)
    if right_location:
        return f"{right_location}_left"

    left_location = has_ramp(results, (bottom_left, upper_left), 0.4, 0.2)
    if left_location:
        return f"{left_location}_right"

    return False

def has_ramp(results, areas, partial_ratio, all_ratio):
    bottom, upper = areas
    input_height, input_width = results.shape

    mask = np.zeros((input_height, input_width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [bottom], 1)
    label, count = np.unique(results[mask == 1], return_counts=True)
    has_ramp_bottom = parse_result(label, count, partial_ratio)

    mask = np.zeros((input_height, input_width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [upper], 1)
    label, count = np.unique(results[mask == 1], return_counts=True)
    has_ramp_upper = parse_result(label, count, partial_ratio)

    if has_ramp_bottom:
        return "bottom"
    if has_ramp_upper:
        return "upper"

    mask = np.zeros((input_height, input_width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [bottom], 1)
    mask = cv2.fillPoly(mask, [upper], 1)
    label, count = np.unique(results[mask == 1], return_counts=True)
    has_ramp = parse_result(label, count, all_ratio)
    if has_ramp:
        return "center"
    else:
        return False

def _draw_closest_and_future(big, small, img_rgb):
    big_closest, big_future = big
    small_closest, small_future = small

    img_array = np.array(img_rgb)
    big_closest_color = [0, 50, 50]
    big_future_color = [0, 69, 0]

    small_closest_color = [0, 100, 100]
    small_future_color = [69, 69, 69]

    height, weight, channel = img_array.shape
    img = np.zeros((height, weight, channel), dtype=np.uint8)
    img = cv2.fillPoly(img, [big_closest], big_closest_color)
    img = cv2.fillPoly(img, [big_future], big_future_color)
    img = cv2.fillPoly(img, [small_closest], small_closest_color)
    img = cv2.fillPoly(img, [small_future], small_future_color)

    img_array = 0.3 * img + img_array

    cv2.imwrite("test.png", img_array)

if __name__ == '__main__':
    web_app = InferenceServer("lifelong-learning-robo", host=get_host_ip())
    web_app.start()

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
import cv2
import time
import requests
import tenacity
from tenacity import retry
import numpy as np
from robosdk.utils.logger import logging

LOGGER = logging.bind(instance="lifelong-learning-robo")


@retry(stop=tenacity.stop_after_attempt(5),
       retry=tenacity.retry_if_result(lambda x: x is None),
       wait=tenacity.wait_fixed(1))
def http_request(url, method=None, timeout=None, binary=True, **kwargs):
    _maxTimeout = timeout if timeout else 10
    _method = "GET" if not method else method
    try:
        response = requests.request(method=_method, url=url, **kwargs)
        if response.status_code == 200:
            return (response.json() if binary else
                    response.content.decode("utf-8"))
        elif 200 < response.status_code < 400:
            LOGGER.info(f"Redirect_URL: {response.url}")
        LOGGER.warning(
            'Get invalid status code %s while request %s',
            response.status_code,
            url)
    except (ConnectionRefusedError, requests.exceptions.ConnectionError):
        LOGGER.warning(f'Connection refused while request {url}')
    except requests.exceptions.HTTPError as err:
        LOGGER.warning(f"Http Error while request {url} : f{err}")
    except requests.exceptions.Timeout as err:
        LOGGER.warning(f"Timeout Error while request {url} : f{err}")
    except requests.exceptions.RequestException as err:
        LOGGER.warning(f"Error occurred while request {url} : f{err}")


class Estimator:
    def __init__(self, service_name="lifelong-learning-robo",
                 input_size=(240, 424)):
        self.input_height, self.input_width = input_size
        self.remote_ip = os.getenv("BIG_MODEL_IP", "http://localhost")
        self.port = int(os.getenv("BIG_MODEL_PORT", "8080"))
        self.endpoint = f"{self.remote_ip}:{self.port}/{service_name}/predict"
        self.curr_gait = ""
        self.fps = 30
        self.inferDangerCount = 0
        self.hold_time = 3
        self.freq = cv2.getTickFrequency()
        self.last_change = int(time.time())
        self._poly = np.array([
            [0, int(self.input_height)],
            [int(self.input_width),
             int(self.input_height)],
            [int(0.764 * self.input_width + .5),
             int(0.865 * self.input_height + .5)],
            [int(0.236 * self.input_width + .5),
             int(0.865 * self.input_height + .5)]
        ], dtype=np.int32)

    def predict(self, rgb, depth):
        t1 = cv2.getTickCount()
        image = cv2.imencode('.jpg', rgb)[1].tobytes()
        depth = cv2.imencode('.jpg', depth)[1].tobytes()
        orig_h, orig_w, _ = rgb.shape
        result = http_request(
            self.endpoint, method="POST", files={
                "image": ('rgb.jpg', image, "image/jpeg"),
                "depth": None
            }
        )
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / self.freq
        self.fps = round(1 / time1, 2)

        if not isinstance(result, dict):
            return

        msg = result.get("msg", "")
        r = result.get("result", {})

        code = int(result.get("code", 1)) if str(
            result.get("code", 1)).isdigit() else 1
        if len(msg) and code != 0:
            LOGGER.warning(msg)
            return

        _type = int(r.get("type", 1))
        if _type == 1:
            self.curr_gait = "stop"
            LOGGER.warning("unknown result")
        step = 1.0 / self.fps

        c = r.get("curr", "unknown")
        f = r.get("future", "unknown")
        if c == "curb" or f == "curb" or c != f:
            self.inferDangerCount = min(self.inferDangerCount + step, 10)
        else:
            self.inferDangerCount = max(self.inferDangerCount - 2 * step, 0)

        if self.inferDangerCount > 1:
            self.curr_gait = "up-stair"
            # self.last_change = int(time.time())
        elif self.inferDangerCount == 0:
            # now = int(time.time())
            # if now - self.last_change > self.hold_time:
            self.curr_gait = "trot"

        location = result.get("result").get("ramp")
        return location

if __name__ == '__main__':
    os.environ["BIG_MODEL_IP"] = "http://100.94.29.220"
    os.environ["BIG_MODEL_PORT"] = "30001"
    f1 = "./E1_door.1716.rgb.png"
    f2 = "./E1_door.1716.depth.png"
    _frame = cv2.imread(f1)
    _depth = cv2.imread(f2, -1) / 1000
    d = Estimator()
    cv2.imwrite("./frame_1716.out.png", d.predict(_frame, _depth))

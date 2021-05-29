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
import json
import time
import threading
from copy import deepcopy
from retrying import retry
from requests import request
import websockets
import asyncio
from websockets.exceptions import InvalidStatusCode, WebSocketException, ConnectionClosedError, ConnectionClosedOK
from sedna.common.log import sednaLogger


@retry(stop_max_attempt_number=3, retry_on_result=lambda x: x is None, wait_fixed=2000)
def http_request(url, method=None, timeout=None, binary=True, **kwargs):
    _maxTimeout = timeout if timeout else 300
    _method = "GET" if not method else method
    try:
        response = request(method=_method, url=url, **kwargs)
        if response.status_code == 200:
            return response.json() if binary else response.content.decode("utf-8")
        elif 200 < response.status_code < 400:
            sednaLogger.info(f"Redirect_URL: {response.url}")
        sednaLogger.error('Get invalid status code %s while request %s', response.status_code, url)
    except Exception as e:
        sednaLogger.error(f'Error occurred while request {url}, Msg: {e}', exc_info=True)


class LCReporter(threading.Thread):
    """Inherited thread, which is an entity that periodically report to
    the lc.
    """

    def __init__(self, message, period_interval=30):
        threading.Thread.__init__(self)

        # the value of statistics
        self.inference_number = 0
        self.hard_example_number = 0
        self.period_interval = period_interval
        # The system resets the period_increment after sending the messages to
        # the LC. If the period_increment is 0 in the current period,
        # the system does not send the messages to the LC.
        self.period_increment = 0
        self.message = deepcopy(message)
        self.lock = threading.Lock()

    def update_for_edge_inference(self):
        self.lock.acquire()
        self.inference_number += 1
        self.period_increment += 1
        self.lock.release()

    def update_for_collaboration_inference(self):
        self.lock.acquire()
        self.inference_number += 1
        self.hard_example_number += 1
        self.period_increment += 1
        self.lock.release()

    def run(self):
        while True:

            start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            time.sleep(self.period_interval)
            if self.period_increment == 0:
                continue

            info = {
                "startTime": start,
                "updateTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "inferenceNumber": self.inference_number,
                "hardExampleNumber": self.hard_example_number,
                "uploadCloudRatio": self.hard_example_number / self.inference_number
            }
            message = deepcopy(self.message)
            message["ownerInfo"] = info
            LCClient.send(message["ownerName"], message["name"], message)
            self.period_increment = 0


class LCClient:

    @classmethod
    def send(cls, lc_server, worker_name, message: dict):
        url = '{0}/sedna/workers/{1}/info'.format(
            lc_server, worker_name
        )
        return http_request(url=url, method="POST", json=message)


class AggregationClient:
    """Client that interacts with the cloud aggregator."""
    _ws_timeout = 5
    _retry = 15
    _retry_interval_seconds = 3
    max_size = 500 * 1024 * 1024

    def __init__(self, url, client_id, **kwargs):
        self.uri = url
        self.client_id = client_id
        self.ws = None
        self.kwargs = kwargs or {}
        timeout = self.kwargs.get("ping_timeout", "")
        timeout = int(timeout) if str(timeout).isdigit() else self._ws_timeout
        interval = self.kwargs.get("ping_interval", "")
        interval = int(interval) if str(interval).isdigit() else timeout * self._retry_interval_seconds
        max_size = self.kwargs.get("max_size", "")
        max_size = int(max_size) if str(max_size).isdigit() else self.max_size
        self.kwargs.update({
            "ping_timeout": timeout,
            "ping_interval": interval,
            "max_size": min(max_size, 16 * 1024 * 1024)
        })

    async def connect(self):
        sednaLogger.info(f"{self.uri} connection by {self.client_id}")

        try:
            conn = websockets.connect(
                self.uri, **self.kwargs
            )
            self.ws = await conn.__aenter__()
            await self.ws.send(json.dumps({'type': 'subscribe', 'client_id': self.client_id}))

            res = await self.ws.recv()
            return res
        except ConnectionRefusedError:
            sednaLogger.info(f"{self.uri} connection was refused by server")
            raise
        except ConnectionClosedError:
            sednaLogger.info(f"{self.uri}  connection lost")
            raise
        except ConnectionClosedOK:
            sednaLogger.info(f"{self.uri}  connection closed")
            raise
        except InvalidStatusCode as err:
            sednaLogger.info(f"{self.uri}  Websocket failed - with invalid status code {err.status_code}")
            raise
        except WebSocketException as err:
            sednaLogger.info(f"{self.uri}  Websocket failed - with {err}")
            raise
        except OSError as err:
            sednaLogger.info(f"{self.uri} Connection failed - with {err}")
            raise
        except Exception:
            sednaLogger.exception(f"{self.uri} Websocket Error")
            raise

    async def _send(self, data):
        error = ""
        for _ in range(self._retry):
            try:
                await self.ws.send(data)
                result = await self.ws.recv()
                return result
            except Exception as e:
                error = e
                sednaLogger.warning(f"send data error: {error}")
                time.sleep(self._retry_interval_seconds)
        sednaLogger.error(f"websocket error: {error}, retry times: {self._retry}")
        return None

    def send(self, data, msg_type="message", job_name=""):
        loop = asyncio.get_event_loop()
        j = json.dumps({
            "type": msg_type, "client": self.client_id,
            "data": data, "job_name": job_name
        })
        data_json = loop.run_until_complete(self._send(j))
        if data_json is None:
            sednaLogger.error("send data to agg worker failed, exist worker")
            return
        res = json.loads(data_json)
        return res


class ModelClient:
    """Remote model service"""

    def __init__(self, service_name, version="", host="127.0.0.1", port="8080", protocol="http"):
        self.server_name = f"{service_name}{version}"
        self.endpoint = f"{protocol}://{host}:{port}/{service_name}"

    def check_server_status(self):
        try:
            http_request(url=self.endpoint, method="GET", json={})
        except:
            return False
        else:
            return True

    def inference(self, x, **kwargs):
        """Use the remote big model server to inference."""
        json_data = deepcopy(kwargs)
        json_data.update({"data": x})
        _url = "{}/{}".format(self.endpoint, "predict")
        try:
            res = http_request(url=_url, method="POST", json=json_data)
        except:
            return None
        return res


class KBClient:
    """Communicate with Knowledge Base server"""

    def __init__(self, kbserver):
        self.kbserver = f"{kbserver}/knowledgebase"

    def upload_file(self, files, name=""):
        if not (files and os.path.isfile(files)):
            return files
        if not name:
            name = os.path.basename(name)
        _url = f"{self.kbserver}/file/upload"
        with open(name, "rb") as fin:
            files = {"file": fin}
            outurl = http_request(url=_url, method="POST", files=files)
        return outurl

    def update_db(self, task_info_file):

        _url = f"{self.kbserver}/update"

        try:
            with open(task_info_file, "rb") as fin:
                files = {"task": fin}
                _id = http_request(url=_url, method="POST", files=files)
        except Exception as err:
            sednaLogger.error(f"Update kb error: {err}")
            _id = None
        return _id

    def update_task_status(self, tasks, new_status=1):
        data = {
            "tasks": tasks,
            "status": int(new_status)
        }
        _url = f"{self.kbserver}/update/status"
        return http_request(url=_url, method="POST", json=data)

    def query_db(self, sample):
        pass

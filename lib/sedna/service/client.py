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
import asyncio
import threading
from copy import deepcopy

import tenacity
from tenacity import retry
import requests
import websockets
from websockets.exceptions import InvalidStatusCode, WebSocketException
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from sedna.common.log import LOGGER
from sedna.common.file_ops import FileOps


@retry(stop=tenacity.stop_after_attempt(5),
       retry=tenacity.retry_if_result(lambda x: x is None),
       wait=tenacity.wait_fixed(3))
def http_request(
    url,
    method=None,
    timeout=None,
    binary=True,
    no_decode=False,
    **kwargs
):

    _maxTimeout = timeout if timeout else 300
    _method = "GET" if not method else method
    try:
        response = requests.request(method=_method, url=url, **kwargs)
        if response.status_code == 200:
            if no_decode:
                return response
            else:
                return (response.json() if binary else
                        response.content.decode("utf-8"))
        elif 200 < response.status_code < 400:
            LOGGER.info(f"Redirect_URL: {response.url}")
        LOGGER.warning(
            f'Get invalid status code \
                {response.status_code} in request {url}')

    except (ConnectionRefusedError, requests.exceptions.ConnectionError):
        LOGGER.warning(f'Connection refused in request {url}')
    except requests.exceptions.HTTPError as err:
        LOGGER.warning(f"Http Error in request {url} : {err}")
    except requests.exceptions.Timeout as err:
        LOGGER.warning(f"Timeout Error in request {url} : {err}")
    except requests.exceptions.RequestException as err:
        LOGGER.warning(f"Error occurred in request {url} : {err}")


class LCReporter(threading.Thread):
    """Inherited thread, which is an entity that periodically report to
    the lc.
    """

    def __init__(self, lc_server, message, period_interval=30):
        threading.Thread.__init__(self)

        # the value of statistics
        self.inference_number = 0
        self.hard_example_number = 0
        self.period_interval = period_interval
        self.lc_server = lc_server
        # The system resets the period_increment after sending the messages to
        # the LC. If the period_increment is 0 in the current period,
        # the system does not send the messages to the LC.
        self.period_increment = 0
        self.message = message
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
                "updateTime": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime()),
                "inferenceNumber": self.inference_number,
                "hardExampleNumber": self.hard_example_number,
                "uploadCloudRatio": (self.hard_example_number /
                                     self.inference_number)
            }
            self.message["ownerInfo"] = info
            LCClient.send(self.lc_server,
                          self.message["name"],
                          self.message)
            self.period_increment = 0


class LCClient:
    """send info to LC by http"""
    @classmethod
    def send(cls, lc_server, worker_name, message: dict):
        url = '{0}/sedna/workers/{1}/info'.format(
            lc_server, worker_name
        )
        return http_request(url=url, method="POST", json=message)


class AggregationClient:
    """Client that interacts with the cloud aggregator."""
    _ws_timeout = 5
    max_size = 500 * 1024 * 1024

    def __init__(self, url, client_id, **kwargs):
        self.uri = url
        self.client_id = client_id
        self.ws = None
        self.kwargs = kwargs or {}
        timeout = self.kwargs.get("ping_timeout", "")
        timeout = int(timeout) if str(timeout).isdigit() else self._ws_timeout
        interval = self.kwargs.get("ping_interval", "")
        interval = int(interval) if str(interval).isdigit(
        ) else self._ws_timeout
        max_size = self.kwargs.get("max_size", "")
        max_size = int(max_size) if str(max_size).isdigit() else self.max_size
        self.kwargs.update({
            "ping_timeout": timeout,
            "ping_interval": interval,
            "max_size": min(max_size, 16 * 1024 * 1024)
        })
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            asyncio.wait_for(self.connect(), timeout=timeout)
        )

    @retry(stop=tenacity.stop_after_attempt(15),
           retry=tenacity.retry_if_result(lambda x: x is None),
           wait=tenacity.wait_fixed(3))
    async def connect(self):
        LOGGER.info(f"{self.uri} connection by {self.client_id}")
        try:
            self.ws = await asyncio.wait_for(websockets.connect(
                self.uri, **self.kwargs
            ), self._ws_timeout)
            await self.ws.send(json.dumps({'type': 'subscribe',
                                           'client_id': self.client_id}))
            res = await self.ws.recv()
            return res
        except ConnectionRefusedError:
            LOGGER.warning(f"{self.uri} connection was refused by server")
        except ConnectionClosedError:
            LOGGER.warning(f"{self.uri} connection lost")
        except ConnectionClosedOK:
            LOGGER.warning(f"{self.uri} connection closed")
        except InvalidStatusCode as err:
            LOGGER.warning(
                f"{self.uri} websocket failed - "
                f"with invalid status code {err.status_code}")
        except WebSocketException as err:
            LOGGER.warning(f"{self.uri} websocket failed - with {err}")
        except OSError as err:
            LOGGER.warning(f"{self.uri} connection failed - with {err}")

    @retry(stop=tenacity.stop_after_attempt(15),
           retry=tenacity.retry_if_result(lambda x: x is None),
           wait=tenacity.wait_fixed(3))
    async def _send(self, data):
        try:
            await asyncio.wait_for(self.ws.send(data), self._ws_timeout)
            return True
        except Exception as err:
            LOGGER.info(f"{self.uri} send data failed - with {err}")

    async def _recv(self):
        result = await self.ws.recv()
        return result

    def send(self, data, msg_type="message", job_name=""):
        loop = asyncio.get_event_loop()
        j = json.dumps({
            "type": msg_type, "client": self.client_id,
            "data": data, "job_name": job_name
        })
        loop.run_until_complete(self._send(j))

    def recv(self, wait_data_type=None):
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(self._recv())
        try:
            data = json.loads(data)
        except (json.decoder.JSONDecodeError, TypeError):
            pass
        if not wait_data_type or (isinstance(data, dict) and
                                  data.get("type", "") == wait_data_type):
            return data


class ModelClient:
    """Remote model service"""

    def __init__(self, service_name, version="",
                 host="127.0.0.1", port="8080", protocol="http"):
        self.server_name = f"{service_name}{version}"
        self.endpoint = f"{protocol}://{host}:{port}/{service_name}"

    def check_server_status(self):
        return http_request(url=self.endpoint, method="GET")

    def inference(self, x, **kwargs):
        """Use the remote big model server to inference."""
        json_data = deepcopy(kwargs)
        json_data.update({"data": x})
        _url = f"{self.endpoint}/predict"
        return http_request(url=_url, method="POST", json=json_data)


class KBClient:
    """Communicate with Knowledge Base server"""

    def __init__(self, kbserver):
        self.kbserver = f"{kbserver}/knowledgebase"

    def upload_file(self, files, name=""):
        if not (files and os.path.isfile(files)):
            return files
        if not name:
            name = os.path.basename(files)
        LOGGER.info(f"Try to upload file {name}")
        _url = f"{self.kbserver}/file/upload"
        with open(files, "rb") as fin:
            files = {"file": fin}
            outurl = http_request(url=_url, method="POST", files=files)
        if FileOps.is_remote(outurl):
            return outurl
        outurl = outurl.lstrip("/")
        FileOps.delete(files)
        return f"{self.kbserver}/{outurl}"

    def update_db(self, task_info_file):

        _url = f"{self.kbserver}/update"

        try:
            with open(task_info_file, "rb") as fin:
                files = {"task": fin}
                outurl = http_request(url=_url, method="POST", files=files)

        except Exception as err:
            LOGGER.error(f"Update kb error: {err}")
            outurl = None
        else:
            if not FileOps.is_remote(outurl):
                outurl = outurl.lstrip("/")
                outurl = f"{self.kbserver}/{outurl}"
            FileOps.delete(task_info_file)
        return outurl

    def update_task_status(self, tasks: str, new_status=1):
        data = {
            "tasks": tasks,
            "status": int(new_status)
        }
        _url = f"{self.kbserver}/update/status"
        try:
            outurl = http_request(url=_url, method="POST", json=data)
        except Exception as err:
            LOGGER.error(f"Update kb error: {err}")
            outurl = None
        if not outurl:
            return None
        if not FileOps.is_remote(outurl):
            outurl = outurl.lstrip("/")
            outurl = f"{self.kbserver}/{outurl}"
        return outurl

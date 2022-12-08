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

import asyncio

import tenacity
import websockets
from websockets.exceptions import InvalidStatusCode, WebSocketException
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from robosdk.utils.constant import MsgChannelItem
from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory
from .base import MSGChannelBase


__all__ = ("WSMessageChannel",)


@ClassFactory.register(ClassType.GENERAL)
class WSMessageChannel(MSGChannelBase):  # noqa

    def __init__(self):
        super(WSMessageChannel, self).__init__()
        self.has_connect = False
        self.ws = None

    def start(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            asyncio.wait_for(
                self.connect(),
                timeout=MsgChannelItem.GET_MESSAGE_TIMEOUT.value
            )
        )
        self.run()

    def run(self):
        while 1:
            data = self.get_data()
            if not data:
                continue
            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(self._send(data))
            except Exception as err:
                self.logger.error(err)
                break
        self.has_connect = False
        self.start()

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(MsgChannelItem.CONNECT_INTERVAL.value),
        retry=tenacity.retry_if_result(lambda x: x is None),
        wait=tenacity.wait_fixed(3))
    async def _send(self, data):
        try:
            await asyncio.wait_for(
                self.ws.send(data),
                MsgChannelItem.CONNECT_INTERVAL.value
            )
            return True
        except Exception as err:
            self.logger.error(f"{self.endpoint} send data failed - with {err}")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(MsgChannelItem.CONNECT_INTERVAL.value),
        retry=tenacity.retry_if_result(lambda x: x is None),
        wait=tenacity.wait_fixed(3))
    async def connect(self):
        self.logger.info(f'connecting to {self.endpoint}')
        ssl_context = {"ssl": self.build_ssl()} if self.endpoint.startswith(
            'wss://') else {}

        try:
            self.ws = await asyncio.wait_for(websockets.connect(
                self.endpoint, **ssl_context
            ), MsgChannelItem.CONNECT_INTERVAL.value)
        except ConnectionRefusedError:
            self.logger.warning(f"{self.endpoint} connection refused by server")
        except ConnectionClosedError:
            self.logger.warning(f"{self.endpoint} connection lost")
        except ConnectionClosedOK:
            self.logger.warning(f"{self.endpoint} connection closed")
        except InvalidStatusCode as err:
            self.logger.warning(
                f"{self.endpoint} websocket failed - "
                f"with invalid status code {err.status_code}")
        except WebSocketException as err:
            self.logger.warning(f"{self.endpoint} websocket failed: {err}")
        except OSError as err:
            self.logger.warning(f"{self.endpoint} connection failed: {err}")
        else:
            self.has_connect = True
            return True

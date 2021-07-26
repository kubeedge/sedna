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

import time
from typing import List, Optional, Dict, Any

import uuid
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket
from fastapi.routing import APIRoute
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import WebSocketRoute
from starlette.endpoints import WebSocketEndpoint
from starlette.types import ASGIApp, Receive, Scope, Send

from sedna.common.log import LOGGER
from sedna.common.utils import get_host_ip
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.algorithms.aggregation import AggClient

from .base import BaseServer

__all__ = ('AggregationServer',)


class WSClientInfo(BaseModel):  # pylint: disable=too-few-public-methods
    """
    client information
    """
    client_id: str
    connected_at: float
    info: Any


class WSClientInfoList(BaseModel):  # pylint: disable=too-few-public-methods
    clients: List


class WSEventMiddleware:  # pylint: disable=too-few-public-methods
    def __init__(self, app: ASGIApp, **kwargs):
        self._app = app
        self._server = Aggregator(**kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] in ("lifespan", "http", "websocket"):
            servername = scope["path"].lstrip("/")
            scope[servername] = self._server
        await self._app(scope, receive, send)
        # exit agg server if job complete
        scope["app"].shutdown = (self._server.exit_check()
                                 and self._server.empty)


class WSServerBase:
    def __init__(self):
        self._clients: Dict[str, WebSocket] = {}
        self._client_meta: Dict[str, WSClientInfo] = {}

    def __len__(self) -> int:
        return len(self._clients)

    @property
    def empty(self) -> bool:
        return len(self._clients) == 0

    @property
    def client_list(self) -> List[str]:
        # Todo: Considering the expansion of server center,
        #  saving the data to a database would be more appropriate.

        return list(self._clients)

    def add_client(self, client_id: str, websocket: WebSocket):
        if client_id in self._clients:
            raise ValueError(f"Client {client_id} is already in the server")
        LOGGER.info(f"Adding client {client_id}")
        self._clients[client_id] = websocket
        self._client_meta[client_id] = WSClientInfo(
            client_id=client_id, connected_at=time.time(), info=None
        )

    async def kick_client(self, client_id: str):
        if client_id not in self._clients:
            raise ValueError(f"Client {client_id} is not in the server")
        await self._clients[client_id].close()

    def remove_client(self, client_id: str):
        if client_id not in self._clients:
            raise ValueError(f"Client {client_id} is not in the server")
        LOGGER.info(f"Removing Client {client_id} from server")
        del self._clients[client_id]
        del self._client_meta[client_id]

    def get_client(self, client_id: str) -> Optional[WSClientInfo]:
        return self._client_meta.get(client_id)

    async def send_message(self, client_id: str, msg: Dict):
        for to_client, websocket in self._clients.items():
            if to_client == client_id:
                continue
            LOGGER.info(f"send data to Client {to_client} from server")
            await websocket.send_json(msg)

    async def client_joined(self, client_id: str):
        for websocket in self._clients.values():
            await websocket.send_json({"type": "CLIENT_JOIN",
                                       "data": client_id})


class Aggregator(WSServerBase):
    def __init__(self, **kwargs):
        super(Aggregator, self).__init__()
        self.exit_round = int(kwargs.get("exit_round", 3))
        aggregation = kwargs.get("aggregation", "FedAvg")
        self.aggregation = ClassFactory.get_cls(ClassType.FL_AGG, aggregation)
        if callable(self.aggregation):
            self.aggregation = self.aggregation()
        self.participants_count = int(kwargs.get("participants_count", "1"))
        self.current_round = 0

    async def send_message(self, client_id: str, msg: Dict):
        data = msg.get("data")
        if data and msg.get("type", "") == "update_weight":
            info = AggClient()
            info.num_samples = int(data["num_samples"])
            info.weights = data["weights"]
            self._client_meta[client_id].info = info
            current_clinets = [
                x.info for x in self._client_meta.values() if x.info
            ]
            # exit while aggregation job is NOT start
            if len(current_clinets) < self.participants_count:
                return
            self.current_round += 1
            weights = self.aggregation.aggregate(current_clinets)
            exit_flag = "ok" if self.exit_check() else "continue"

            msg["type"] = "recv_weight"
            msg["round_number"] = self.current_round
            msg["data"] = {
                "total_sample": self.aggregation.total_size,
                "round_number": self.current_round,
                "weights": weights,
                "exit_flag": exit_flag
            }
        for to_client, websocket in self._clients.items():
            try:
                await websocket.send_json(msg)
            except Exception as err:
                LOGGER.error(err)
            else:
                if msg["type"] == "recv_weight":
                    self._client_meta[to_client].info = None

    def exit_check(self):
        return self.current_round >= self.exit_round


class BroadcastWs(WebSocketEndpoint):
    encoding: str = "json"
    session_name: str = ""
    count: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server: Optional[Aggregator] = None
        self.client_id: Optional[str] = None

    async def on_connect(self, websocket: WebSocket):
        servername = websocket.scope['path'].lstrip("/")
        LOGGER.info("Connecting new client...")
        server: Optional[Aggregator] = self.scope.get(servername)
        if server is None:
            raise RuntimeError("HOST `client` instance unavailable!")
        self.server = server
        await websocket.accept()

    async def on_disconnect(self, _websocket: WebSocket, _close_code: int):
        if self.client_id is None:
            raise RuntimeError(
                "on_disconnect() called without a valid client_id"
            )
        self.server.remove_client(self.client_id)

    async def on_receive(self, _websocket: WebSocket, msg: Dict):
        command = msg.get("type", "")
        if command == "subscribe":
            self.client_id = msg.get("client_id", "") or uuid.uuid4().hex
            await self.server.client_joined(self.client_id)
            self.server.add_client(self.client_id, _websocket)
        if self.client_id is None:
            raise RuntimeError(
                "on_receive() called without a valid client_id")
        await self.server.send_message(self.client_id, msg)


class AggregationServer(BaseServer):
    def __init__(
            self,
            aggregation: str,
            host: str = None,
            http_port: int = 7363,
            exit_round: int = 1,
            participants_count: int = 1,
            ws_size: int = 10 * 1024 * 1024):
        if not host:
            host = get_host_ip()
        super(
            AggregationServer,
            self).__init__(
            servername=aggregation,
            host=host,
            http_port=http_port,
            ws_size=ws_size)
        self.aggregation = aggregation
        self.participants_count = participants_count
        self.exit_round = max(int(exit_round), 1)
        self.app = FastAPI(
            routes=[
                APIRoute(
                    f"/{aggregation}",
                    self.client_info,
                    response_class=JSONResponse,
                ),
                WebSocketRoute(
                    f"/{aggregation}",
                    BroadcastWs
                )
            ],
        )
        self.app.shutdown = False

    def start(self):
        """
        Start the server
        """

        self.app.add_middleware(
            WSEventMiddleware,
            exit_round=self.exit_round,
            aggregation=self.aggregation,
            participants_count=self.participants_count
        )  # define the aggregation method and exit condition

        self.run(self.app, ws_max_size=self.ws_size)

    async def client_info(self, request: Request):
        server: Optional[Aggregator] = request.get(self.server_name)
        try:
            data = await request.json()
        except BaseException:
            data = {}
        client_id = data.get("client_id", "") if data else ""
        if client_id:
            return server.get_client(client_id)
        return WSClientInfoList(clients=server.client_list)

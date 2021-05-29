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
import uuid
import time
from typing import List, Optional, Dict
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket
from fastapi.routing import APIRoute
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import WebSocketRoute
from starlette.endpoints import WebSocketEndpoint
from starlette.types import ASGIApp, Receive, Scope, Send
from sedna.common.log import sednaLogger
from sedna.common.utils import get_host_ip

from .base import BaseServer


__all__ = ('AggregationServer',)


class WSClientInfo(BaseModel):  # pylint: disable=too-few-public-methods
    """
    client information
    """

    client_id: str
    connected_at: float
    job_count: int


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
        sednaLogger.info(f"Adding client {client_id}")
        self._clients[client_id] = websocket
        self._client_meta[client_id] = WSClientInfo(
            client_id=client_id, connected_at=time.time(), job_count=0
        )

    async def kick_client(self, client_id: str):
        if client_id not in self._clients:
            raise ValueError(f"Client {client_id} is not in the server")
        await self._clients[client_id].close()

    def remove_client(self, client_id: str):
        if client_id not in self._clients:
            raise ValueError(f"Client {client_id} is not in the server")
        sednaLogger.info(f"Removing Client {client_id} from server")
        del self._clients[client_id]
        del self._client_meta[client_id]

    def get_client(self, client_id: str) -> Optional[WSClientInfo]:
        return self._client_meta.get(client_id)

    async def send_message(self, client_id: str, msg: Dict):
        self._client_meta[client_id].job_count += 1
        for to_client, websocket in self._clients.items():
            if to_client == client_id:
                continue
            sednaLogger.info(f"send data to Client {to_client} from server")
            await websocket.send_json(msg)

    async def client_joined(self, client_id: str):
        for websocket in self._clients.values():
            await websocket.send_json({"type": "CLIENT_JOIN", "data": client_id})

    async def client_left(self, client_id: str):
        for to_client, websocket in self._clients.items():
            if to_client == client_id:
                continue
            await websocket.send_json({"type": "CLIENT_LEAVE", "data": client_id})


class Aggregator(WSServerBase):
    def __init__(self, **kwargs):
        super(Aggregator, self).__init__()
        self.exit_round = int(kwargs.get("exit_round", 3))
        self.current_round = {}

    async def send_message(self, client_id: str, msg: Dict):
        self._client_meta[client_id].job_count += 1
        for to_client, websocket in self._clients.items():
            if to_client == client_id:
                continue
            if msg.get("type", "") == "update_weight":
                self.current_round[client_id] = self.current_round.get(client_id, 0) + 1
                exit_flag = "ok" if self.exit_check(client_id) else "continue"
                msg["exit_flag"] = exit_flag
            await websocket.send_json(msg)

    def exit_check(self, client_id):
        current_round = self.current_round.get(client_id, 0)
        return current_round >= self.exit_round


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
        sednaLogger.info("Connecting new client...")
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
        await self.server.client_left(self.client_id)

    async def on_receive(self, _websocket: WebSocket, msg: Dict):
        command = msg.get("type", "")
        if command == "subscribe":
            self.client_id = msg.get("client_id", "") or uuid.uuid4().hex
            await self.server.client_joined(self.client_id)
            self.server.add_client(self.client_id, _websocket)
        if self.client_id is None:
            raise RuntimeError("on_receive() called without a valid client_id")
        await self.server.send_message(self.client_id, msg)


class AggregationServer(BaseServer):
    def __init__(self, servername: str, host: str = None, http_port: int = 7363, exit_round: int = 1):
        if not host:
            host = get_host_ip()
        super(AggregationServer, self).__init__(servername=servername, host=host,
                                                http_port=http_port)
        self.server_name = servername
        self.exit_round = max(int(exit_round), 1)
        self.app = FastAPI(
            routes=[
                APIRoute(
                    f"/{servername}",
                    self.client_info,
                    response_class=JSONResponse,
                ),
                WebSocketRoute(
                    f"/{servername}",
                    BroadcastWs
                )
            ],
        )

    def start(self):
        """
        Start the server
        """
        self.app.add_middleware(WSEventMiddleware, exit_round=self.exit_round)
        self.run(self.app, websocket_max_message_size=self.ws_size)

    async def client_info(self, request: Request):
        server: Optional[Aggregator] = request.get(self.server_name)
        try:
            data = await request.json()
        except:
            data = {}
        client_id = data.get("client_id", "") if data else ""
        if client_id:
            return server.get_client(client_id)
        return WSClientInfoList(clients=server.client_list)

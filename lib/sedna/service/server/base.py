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

import contextlib
import time
import threading

import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from sedna.common.log import LOGGER
from sedna.common.utils import get_host_ip


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        try:
            yield thread
        finally:
            self.should_exit = True
            thread.join()


class BaseServer:
    # pylint: disable=too-many-instance-attributes,too-many-arguments
    DEBUG = True
    WAIT_TIME = 15

    def __init__(
            self,
            servername: str,
            host: str = '',
            http_port: int = 8080,
            grpc_port: int = 8081,
            workers: int = 1,
            ws_size: int = 16 *
            1024 *
            1024,
            ssl_key=None,
            ssl_cert=None,
            timeout=300):
        self.server_name = servername
        self.app = None
        self.host = host or get_host_ip()
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

        LOGGER.info(f"Start {self.server_name} server over {self.url}")

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.http_port,
            ssl_keyfile=self.keyfile,
            ssl_certfile=self.certfile,
            workers=self.workers,
            timeout_keep_alive=self.timeout,
            log_level="info",
            **kwargs)
        server = Server(config=config)
        with server.run_in_thread() as current_thread:
            return self.wait_stop(current=current_thread)

    def wait_stop(self, current):
        """wait the stop flag to shutdown the server"""
        while 1:
            time.sleep(self.WAIT_TIME)
            if not current.is_alive():
                return
            if getattr(self.app, "shutdown", False):
                return

    def get_all_urls(self):
        url_list = [{"path": route.path, "name": route.name}
                    for route in getattr(self.app, 'routes', [])]
        return url_list

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
import threading

from robosdk.utils.util import Config
from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory

from .base import TaskAgent


@ClassFactory.register(ClassType.GENERAL)
class HttpTaskService(TaskAgent):  # noqa

    def __init__(self,
                 name: str, max_task: int = 100,
                 host: str = "0.0.0.0",
                 port: int = 8080):
        super(HttpTaskService, self).__init__(name=name, max_task=max_task)

        import uvicorn
        from starlette.responses import JSONResponse
        from fastapi import FastAPI
        from fastapi.routing import APIRoute
        from fastapi.middleware.cors import CORSMiddleware

        self.agent_name = self.agent_name.strip("/")
        self._app = FastAPI(
            title=self.agent_name.strip("/"),
            root_path=f"/{self.agent_name}",
            routes=[
                APIRoute(
                    f"/{self.agent_name}/info",
                    self.get_all_url,
                    response_class=JSONResponse,
                    methods=["GET"],
                ),
                APIRoute(
                    f"{self.agent_name}/list",
                    self.get_all_task,
                    response_class=JSONResponse,
                    methods=["GET"],
                ),
                APIRoute(
                    f"{self.agent_name}/create",
                    self.create_task,
                    response_class=JSONResponse,
                    methods=["PUT"],
                ),
                APIRoute(
                    f"{self.agent_name}/state",
                    self.get_task_state,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"{self.agent_name}/delete",
                    self.delete_task,
                    response_class=JSONResponse,
                    methods=["DELETE"],
                ),
            ]
        )
        self._app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"]
        )
        app_conf = uvicorn.Config(
            app=self._app,
            host=host,
            port=port,
            log_level="error"
        )
        self.server = uvicorn.Server(config=app_conf)
        self._run_thread = None

    def start(self):
        self._run_thread = threading.Thread(
            target=self._start_service, daemon=True)
        self._run_thread.start()

    def stop(self):
        self.server.should_exit = True
        # self.server.shutdown()
        self._run_thread.join()

    def _start_service(self):
        self.logger.info(f"starting task service {self.agent_name}")
        self.server.run()
        self.logger.info(f"shutdown task service {self.agent_name}")

    def get_all_url(self):
        return [
            {"path": route.path, "name": route.name} for route in
            getattr(self._app, "routes", [])
        ]

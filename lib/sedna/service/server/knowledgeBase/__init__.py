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
import json
import joblib
import tempfile
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from fastapi.responses import FileResponse
from sedna.service.server.base import BaseServer
from sedna.common.file_ops import FileOps
from sqlalchemy.orm import Session
from .model import *


class KBUpdateResult(BaseModel):  # pylint: disable=too-few-public-methods
    """
    result
    """
    status: bool
    result: Optional[str] = None


class TaskItem(BaseModel):  # pylint: disable=too-few-public-methods
    tasks: List


class KBServer(BaseServer):
    def __init__(self, host: str, http_port: int = 8080, workers: int = 1, save_dir=""):
        servername = "knowledgebase"

        super(KBServer, self).__init__(servername=servername, host=host,
                                       http_port=http_port, workers=workers)
        self.save_dir = FileOps.clean_folder([save_dir], clean=False)[0]
        self.app = FastAPI(
            routes=[
                APIRoute(
                    f"/{servername}/update",
                    self.update,
                    response_model=KBUpdateResult,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{servername}/query",
                    self.query,
                    response_model=TaskItem,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
            ],
            log_level="trace",
            timeout=600,
        )

    def start(self):
        return self.run(self.app)

    def query(self):
        pass

    async def update(self, task: UploadFile = File(...)):
        tasks = await task.read()
        fd, name = tempfile.mkstemp()
        with open(name, "wb") as fout:
            fout.write(tasks)
        task_obj = joblib.load(name)
        with Session(bind=engine) as session:
            for task_group in task_obj:
                grp, g_create = get_or_create(session=session, model=TaskGrp, name=task_group.entry)
                if g_create:

                    grp.sample_num = 0
                    grp.task_num = 0
                    session.add(grp)
                grp.sample_num += len(task_group.samples)
                grp.task_num += len(task_group.tasks)
                t_id = []
                for task in task_group.tasks:
                    t_obj, t_create = get_or_create(session=session, model=Tasks, name=task.entry)
                    if task.meta_attr:
                        t_obj.task_attr = json.dumps(task.meta_attr)
                    if t_create:
                        session.add(t_obj)

                    sampel_obj = Samples(
                        data_type=task.samples.data_type,
                        sample_num=len(task.samples)
                    )
                    session.add(sampel_obj)

                    session.flush()
                    session.commit()
                    sample_dir = FileOps.join_path(self.save_dir,
                                                   f"{sampel_obj.data_type}_{sampel_obj.id}.pkl")
                    task.samples.save(sample_dir)
                    sampel_obj.data_url = sample_dir

                    tsample = TaskSample(sample=sampel_obj, task=t_obj)
                    session.add(tsample)
                    session.flush()
                    t_id.append(t_obj.id)

                model_obj, m_create = get_or_create(session=session, model=TaskModel, task=grp)
                model_obj.model_url = task_group.model.model
                model_obj.is_current = False
                if m_create:
                    session.add(model_obj)
                session.flush()
                session.commit()
                transfer_radio = 1 / grp.task_num
                for t in t_id:
                    t_obj, t_create = get_or_create(session=session, model=TaskRelation, task_id=t, grp=grp)
                    t_obj.transfer_radio = transfer_radio
                    if t_create:
                        session.add(t_obj)
                        session.flush()
                    session.commit()
                session.query(TaskRelation).filter(TaskRelation.grp == grp).update(
                    {"transfer_radio": transfer_radio}
                )

            session.commit()


if __name__ == '__main__':
    KBServer(host="127.0.0.1").start()

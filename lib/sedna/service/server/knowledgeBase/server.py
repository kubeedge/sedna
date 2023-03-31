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
import tempfile
from typing import List, Optional

import joblib
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.routing import APIRoute
from fastapi.responses import FileResponse
from starlette.responses import JSONResponse

from sedna.service.server.base import BaseServer
from sedna.common.file_ops import FileOps
from sedna.common.constant import KBResourceConstant
from sedna.common.config import Context

from .model import *


class KBUpdateResult(BaseModel):  # pylint: disable=too-few-public-methods
    """
    result
    """
    status: int
    tasks: Optional[str] = None


class TaskItem(BaseModel):  # pylint: disable=too-few-public-methods
    tasks: List


class KBServer(BaseServer):
    """
    As knowledge base stored in sqlite, this class realizes creation,
    update and query of the sqlite.
    """
    def __init__(self, host: str, http_port: int = 8080,
                 workers: int = 1, save_dir=""):
        servername = "knowledgebase"

        super(KBServer, self).__init__(servername=servername, host=host,
                                       http_port=http_port, workers=workers)
        self.save_dir = FileOps.clean_folder([save_dir], clean=False)[0]
        self.url = f"{self.url}/{servername}"
        self.kb_index = KBResourceConstant.KB_INDEX_NAME.value
        self.seen_task_key = KBResourceConstant.SEEN_TASK.value
        self.unseen_task_key = KBResourceConstant.UNSEEN_TASK.value
        self.task_group_key = KBResourceConstant.TASK_GROUPS.value
        self.extractor_key = KBResourceConstant.EXTRACTOR.value
        self.app = FastAPI(
            routes=[
                APIRoute(
                    f"/{servername}/update",
                    self.update,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{servername}/update/status",
                    self.update_status,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{servername}/query",
                    self.query,
                    response_model=TaskItem,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    f"/{servername}/file/download",
                    self.file_download,
                    methods=["GET"],
                ),
                APIRoute(
                    f"/{servername}/file/upload",
                    self.file_upload,
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

    def _get_db_index(self):
        _index_path = FileOps.join_path(self.save_dir, self.kb_index)
        if not FileOps.exists(_index_path):  # todo: get from kb
            pass
        return _index_path

    @staticmethod
    def _file_endpoint(files, name=""):
        if not (files and os.path.isfile(files)):
            return files
        filename = name or os.path.basename(files)
        return FileResponse(files, filename=filename)

    async def file_download(self, files: str, name: str = ""):
        files = FileOps.join_path(self.save_dir, files)
        return self._file_endpoint(files, name=name)

    async def file_upload(self, file: UploadFile = File(...)):
        files = await file.read()
        filename = str(file.filename)
        output = FileOps.join_path(self.save_dir, filename)
        with open(output, "wb") as fout:
            fout.write(files)
        return f"/file/download?files={filename}&name={filename}"

    def update_status(self, data: KBUpdateResult = Body(...)):
        deploy = bool(data.status)
        tasks = data.tasks.split(",") if data.tasks else []
        with Session(bind=engine) as session:
            session.query(TaskGrp).filter(
                TaskGrp.name.in_(tasks)
            ).update({
                TaskGrp.deploy: deploy
            }, synchronize_session=False)

        # todo: get from kb
        _index_path = FileOps.join_path(self.save_dir, self.kb_index)
        task_info = FileOps.load(_index_path)
        new_task_group = []

        # TODO: to fit seen tasks and unseen tasks
        default_task = task_info[self.seen_task_key][self.task_group_key][0]
        # todo: get from transfer learning
        for task_group in task_info[self.seen_task_key][self.task_group_key]:
            if not ((task_group.entry in tasks) == deploy):
                new_task_group.append(default_task)
                continue
            new_task_group.append(task_group)
        task_info[self.seen_task_key][self.task_group_key] = new_task_group

        _index_path = FileOps.join_path(self.save_dir, self.kb_index)
        FileOps.dump(task_info, _index_path)
        return f"/file/download?files={self.kb_index}&name={self.kb_index}"

    def update(self, task: UploadFile = File(...)):
        tasks = task.file.read()
        fd, name = tempfile.mkstemp()
        with open(name, "wb") as fout:
            fout.write(tasks)
        os.close(fd)
        upload_info = joblib.load(name)
        # TODO: to adapt unseen tasks
        task_groups = upload_info[self.seen_task_key][self.task_group_key]
        task_groups.extend(
            upload_info[self.unseen_task_key][self.task_group_key])

        with Session(bind=engine) as session:
            # TODO: to adapt unseen tasks
            for task_group in task_groups:
                grp, g_create = get_or_create(
                    session=session, model=TaskGrp, name=task_group.entry)
                if g_create:
                    grp.sample_num = 0
                    grp.task_num = 0
                    session.add(grp)
                grp.sample_num += len(task_group.samples)
                grp.task_num += len(task_group.tasks)
                t_id = []
                for task in task_group.tasks:
                    t_obj, t_create = get_or_create(
                        session=session, model=Tasks, name=task.entry)
                    if task.meta_attr:
                        t_obj.task_attr = json.dumps(task.meta_attr)
                    if t_create:
                        session.add(t_obj)

                    sample_obj = Samples(
                        data_type=task.samples.data_type,
                        sample_num=len(task.samples),
                        data_url=getattr(task, 'data_url', '')
                    )
                    session.add(sample_obj)

                    session.flush()
                    session.commit()
                    tsample = TaskSample(sample=sample_obj, task=t_obj)
                    session.add(tsample)
                    session.flush()
                    t_id.append(t_obj.id)

                model_obj, m_create = get_or_create(
                    session=session, model=TaskModel, task=grp)
                model_obj.model_url = task_group.model.model
                model_obj.is_current = False
                if m_create:
                    session.add(model_obj)
                session.flush()
                session.commit()
                transfer_radio = 1 / grp.task_num
                for t in t_id:
                    t_obj, t_create = get_or_create(
                        session=session, model=TaskRelation,
                        task_id=t, grp=grp)
                    t_obj.transfer_radio = transfer_radio
                    if t_create:
                        session.add(t_obj)
                        session.flush()
                    session.commit()
                session.query(TaskRelation).filter(
                    TaskRelation.grp == grp).update(
                    {"transfer_radio": transfer_radio})

            session.commit()

        # todo: get from kb
        _index_path = FileOps.join_path(self.save_dir, self.kb_index)
        _index_path = FileOps.dump(upload_info, _index_path)

        return f"/file/download?files={self.kb_index}&name={self.kb_index}"

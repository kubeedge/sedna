# Copyright 2023 The KubeEdge Authors.
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
import time
import tempfile

from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.file_ops import FileOps
from sedna.common.config import Context
from sedna.common.constant import KBResourceConstant

from .base_knowledge_management import BaseKnowledgeManagement

__all__ = ('CloudKnowledgeManagement', )


@ClassFactory.register(ClassType.KM)
class CloudKnowledgeManagement(BaseKnowledgeManagement):
    """
    Manage task processing, kb update and task deployment, etc., at cloud.
    """

    def __init__(self, config, seen_estimator, unseen_estimator, **kwargs):
        super(CloudKnowledgeManagement, self).__init__(
            config, seen_estimator, unseen_estimator)

        self.last_task_index = kwargs.get("last_task_index", None)
        self.cloud_output_url = config.get(
            "cloud_output_url", "/tmp")
        self.task_index = FileOps.join_path(
            self.cloud_output_url, config["task_index"])
        self.local_task_index_url = KBResourceConstant.KB_INDEX_NAME.value

    def update_kb(self, task_index):
        if isinstance(task_index, str):
            task_index = FileOps.load(task_index)

        seen_task_index = task_index.get(self.seen_task_key)
        unseen_task_index = task_index.get(self.unseen_task_key)

        seen_extractor, seen_task_groups = self.save_task_index(
            seen_task_index, task_type=self.seen_task_key)
        unseen_extractor, unseen_task_groups = self.save_task_index(
            unseen_task_index, task_type=self.unseen_task_key)

        task_info = {
            self.seen_task_key: {
                self.task_group_key: seen_task_groups,
                self.extractor_key: seen_extractor
            },
            self.unseen_task_key: {
                self.task_group_key: unseen_task_groups,
                self.extractor_key: unseen_extractor
            },
            "create_time": str(time.time())
        }

        fd, name = tempfile.mkstemp()
        FileOps.dump(task_info, name)

        index_file = self.kb_client.update_db(name)
        if not index_file:
            self.log.error("KB update Fail !")
            index_file = name

        return FileOps.upload(index_file, self.task_index)

    def save_task_index(self, task_index, task_type="seen_task"):
        extractor = task_index[self.extractor_key]
        if isinstance(extractor, str):
            extractor = FileOps.load(extractor)
        task_groups = task_index[self.task_group_key]

        model_upload_key = {}
        for task_group in task_groups:
            model_file = task_group.model.model
            save_model = FileOps.join_path(
                self.cloud_output_url, task_type,
                os.path.basename(model_file)
            )
            if model_file not in model_upload_key:
                model_upload_key[model_file] = FileOps.upload(
                    model_file, save_model)

            model_file = model_upload_key[model_file]

            try:
                model = self.kb_client.upload_file(save_model)
                self.log.info(
                    f"Upload task model to {model} successfully."
                )
            except Exception as err:
                self.log.error(
                    f"Upload task model of {model_file} fail: {err}"
                )
                model = FileOps.join_path(
                    self.cloud_output_url,
                    task_type,
                    os.path.basename(model_file))

            task_group.model.model = model

            for _task in task_group.tasks:
                _task.model = FileOps.join_path(
                    self.cloud_output_url,
                    task_type,
                    os.path.basename(model_file))
                sample_dir = FileOps.join_path(
                    self.cloud_output_url, task_type,
                    f"{_task.samples.data_type}_{_task.entry}.sample")
                task_group.samples.save(sample_dir)

                try:
                    sample_dir = self.kb_client.upload_file(sample_dir)
                    self.log.info(
                        f"Upload task sample to {sample_dir} successfully.")
                except Exception as err:
                    self.log.error(
                        f"Upload task samples of {_task.entry} fail: {err}")
                _task.samples.data_url = sample_dir

        save_extractor = FileOps.join_path(
            self.cloud_output_url, task_type,
            f"{task_type}_{KBResourceConstant.TASK_EXTRACTOR_NAME.value}"
        )
        extractor = FileOps.dump(extractor, save_extractor)
        try:
            extractor = self.kb_client.upload_file(extractor)
            self.log.info(
                f"Upload task extractor to {extractor} successfully.")
        except Exception as err:
            self.log.error(f"Upload task extractor fail: {err}")

        return extractor, task_groups

    def evaluate_tasks(self, tasks_detail, **kwargs):
        """
        Parameters
        ----------
        tasks_detail: List[Task]
            output of module task_update_decision,
            consisting of results of evaluation.

        Returns
        -------
        drop_task: List[str]
            names of the tasks which will not to be deployed to the edge.
        """

        self.model_filter_operator = Context.get_parameters("operator", ">")
        self.model_threshold = float(
            Context.get_parameters(
                "model_threshold", 0.1))

        drop_tasks = []

        operator_map = {
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            "=": lambda x, y: x == y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }
        if self.model_filter_operator not in operator_map:
            self.log.warn(
                f"operator {self.model_filter_operator} use to "
                f"compare is not allow, set to <"
            )
            self.model_filter_operator = "<"
        operator_func = operator_map[self.model_filter_operator]

        for detail in tasks_detail:
            scores = detail.scores
            entry = detail.entry
            self.log.info(f"{entry} scores: {scores}")
            if any(map(lambda x: operator_func(float(x),
                                               self.model_threshold),
                       scores.values())):
                self.log.warn(
                    f"{entry} will not be deploy because all "
                    f"scores {self.model_filter_operator} "
                    f"{self.model_threshold}")
                drop_tasks.append(entry)
                continue
        drop_task = ",".join(drop_tasks)

        return drop_task

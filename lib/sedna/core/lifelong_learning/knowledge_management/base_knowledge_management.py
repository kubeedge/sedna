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

from sedna.backend import set_backend
from sedna.common.log import LOGGER
from sedna.common.constant import KBResourceConstant
from sedna.common.config import BaseConfig
from sedna.service.client import KBClient


class BaseKnowledgeManagement:
    """
    Base class of knowledge management.
    It includes model and sample update to knowledge base server.

    Parameters:
    config: BaseConfig, see 'sedna.common.config.BaseConfig' for more details.
        It sets basic configs for knowledge management.
    seen_estimator: Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for a model.
    unseen_estimator: Instance
        An instance with the high-level API that greatly simplifies mechanism
        model learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for a mechanism model.
    """

    def __init__(self, config, seen_estimator, unseen_estimator):
        self.config = BaseConfig()
        if config:
            self.config.from_json(config)

        self.seen_estimator = set_backend(
            estimator=seen_estimator, config=self.config)
        self.unseen_estimator = set_backend(
            estimator=unseen_estimator, config=self.config)
        self.log = LOGGER
        self.seen_task_key = KBResourceConstant.SEEN_TASK.value
        self.unseen_task_key = KBResourceConstant.UNSEEN_TASK.value
        self.task_group_key = KBResourceConstant.TASK_GROUPS.value
        self.extractor_key = KBResourceConstant.EXTRACTOR.value

        self.kb_client = KBClient(kbserver=self.config.ll_kb_server)

    def update_kb(self, task_index):
        raise NotImplementedError

    def save_task_index(self, task_index, task_type=None, **kwargs):
        raise NotImplementedError

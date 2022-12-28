from sedna.backend import set_backend
from sedna.common.log import LOGGER
from sedna.common.constant import KBResourceConstant
from sedna.common.config import BaseConfig
from sedna.service.client import KBClient


class BaseKnowledgeManagement:

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

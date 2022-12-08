import os
import time
import tempfile

from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.file_ops import FileOps
from sedna.common.constant import KBResourceConstant

from .base_knowledge_management import BaseKnowledgeManagement

__all__ = ('CloudKnowledgeManagement', )

@ClassFactory.register(ClassType.KM)
class CloudKnowledgeManagement(BaseKnowledgeManagement):
    """
    Manage task processing, kb update and task deployment, etc., at cloud.

    Parameters:
        ----------
    config: Dict
        parameters to initialize an object
    """

    def __init__(self, config, seen_estimator, unseen_estimator, **kwargs):
        super(CloudKnowledgeManagement, self).__init__(config, seen_estimator, unseen_estimator)

        self.last_task_index = kwargs.get("last_task_index", None)
        self.cloud_output_url = config.get(
            "cloud_output_url", "/tmp")
        self.task_index = FileOps.join_path(
            self.cloud_output_url, config["task_index"])
        self.local_task_index_url = KBResourceConstant.KB_INDEX_NAME.value

        task_evaluation = kwargs.get("task_evaluation") or {}
        self.task_evaluation = task_evaluation.get(
            "method", "TaskEvaluationDefault")
        self.task_evaluation_param = task_evaluation.get("param", {})

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

        index_file = self.kb_server.update_db(name)
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
                model = self.kb_server.upload_file(save_model)
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
                    self.cloud_output_url, task_type, os.path.basename(model_file))
                sample_dir = FileOps.join_path(
                    self.cloud_output_url, task_type,
                    f"{_task.samples.data_type}_{_task.entry}.sample")
                task_group.samples.save(sample_dir)

                try:
                    sample_dir = self.kb_server.upload_file(sample_dir)
                    self.log.info(f"Upload task sample to {sample_dir} successfully.")
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
            extractor = self.kb_server.upload_file(extractor)
            self.log.info(f"Upload task extractor to {extractor} successfully.")
        except Exception as err:
            self.log.error(f"Upload task extractor fail: {err}")

        return extractor, task_groups

    def evaluate_tasks(self, tasks_detail, **kwargs):
        method_cls = ClassFactory.get_cls(
            ClassType.KM, self.task_evaluation)(**self.task_evaluation_param)
        return method_cls(tasks_detail, **kwargs)

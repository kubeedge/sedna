import os
import time
import tempfile

from sedna.common.log import LOGGER
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.file_ops import FileOps
from sedna.common.constant import KBResourceConstant

__all__ = ('CloudKnowledgeManagement', )


@ClassFactory.register(ClassType.KM)
class CloudKnowledgeManagement:
    """
    Manage task processing, kb update and task deployment, etc., at cloud.

    Parameters:
        ----------
    config: Dict
        parameters to initialize an object
    """

    def __init__(self, config, **kwargs):
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

        self.estimator = kwargs.get("estimator") or None
        self.log = LOGGER

        self.seen_task_key = KBResourceConstant.SEEN_TASK.value
        self.unseen_task_key = KBResourceConstant.UNSEEN_TASK.value
        self.task_group_key = KBResourceConstant.TASK_GROUPS.value
        self.extractor_key = KBResourceConstant.EXTRACTOR.value

    def update_kb(self, task_index, kb_server):
        if isinstance(task_index, str):
            task_index = FileOps.load(task_index)

        seen_task_index = task_index.get(self.seen_task_key)
        unseen_task_index = task_index.get(self.unseen_task_key)

        seen_extractor, seen_task_groups = self._save_task_index(
            seen_task_index, kb_server, task_type=self.seen_task_key)
        unseen_extractor, unseen_task_groups = self._save_task_index(
            unseen_task_index, kb_server, task_type=self.unseen_task_key)

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

        index_file = kb_server.update_db(name)
        if not index_file:
            self.log.error("KB update Fail !")
            index_file = name

        return FileOps.upload(index_file, self.task_index)

    def _save_task_index(self, task_index, kb_server, task_type="seen_task"):
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
                model = kb_server.upload_file(save_model)
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
                    sample_dir = kb_server.upload_file(sample_dir)
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
            extractor = kb_server.upload_file(extractor)
        except Exception as err:
            self.log.error(f"Upload task extractor fail: {err}")

        return extractor, task_groups

    def evaluate_tasks(self, tasks_detail, **kwargs):
        method_cls = ClassFactory.get_cls(
            ClassType.KM, self.task_evaluation)(**self.task_evaluation_param)
        return method_cls(tasks_detail, **kwargs)

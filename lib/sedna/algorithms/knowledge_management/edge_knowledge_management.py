import os
import time
import tempfile
import threading

from sedna.common.log import LOGGER
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.file_ops import FileOps
from sedna.common.constant import KBResourceConstant, K8sResourceKindStatus

__all__ = ('EdgeKnowledgeManagement',)


@ClassFactory.register(ClassType.KM)
class EdgeKnowledgeManagement:
    """
    Manage task processing at the edge.

    Parameters:
        ----------
    config: Dict
        parameters to initialize an object
    estimator: Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    """

    def __init__(self, config, estimator, **kwargs):
        self.edge_output_url = config.get(
            "edge_output_url") or "/tmp/edge_output_url"
        self.task_index = FileOps.join_path(
            self.edge_output_url, config["task_index"])
        self.estimator = estimator
        self.log = LOGGER

        self.seen_task_key = KBResourceConstant.SEEN_TASK.value
        self.unseen_task_key = KBResourceConstant.UNSEEN_TASK.value
        self.task_group_key = KBResourceConstant.TASK_GROUPS.value
        self.extractor_key = KBResourceConstant.EXTRACTOR.value

        ModelLoadingThread(self).start()

    def update_kb(self, task_index_url):
        if isinstance(task_index_url, str):
            try:
                task_index = FileOps.load(task_index_url)
            except Exception as err:
                self.log.error(f"{err}")
                self.log.error(
                    "Load task index failed. KB deployment to the edge failed.")
                return None
        else:
            task_index = task_index_url

        FileOps.clean_folder(self.edge_output_url)
        seen_task_index = task_index.get(self.seen_task_key)
        unseen_task_index = task_index.get(self.unseen_task_key)

        seen_extractor, seen_task_groups = self._save_task_index(
            seen_task_index, task_type=self.seen_task_key)
        unseen_extractor, unseen_task_groups = self._save_task_index(
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
            "created_time": task_index.get("created_time", str(time.time()))
        }

        fd, name = tempfile.mkstemp()
        FileOps.dump(task_info, name)
        return FileOps.upload(name, self.task_index)

    def _save_task_index(self, task_index, task_type="seen_task"):
        extractor = task_index[self.extractor_key]
        if isinstance(extractor, str):
            extractor = FileOps.load(extractor)
        task_groups = task_index[self.task_group_key]

        model_upload_key = {}
        for task in task_groups:
            model_file = task.model.model
            save_model = FileOps.join_path(
                self.edge_output_url, task_type,
                os.path.basename(model_file)
            )
            if model_file not in model_upload_key:
                model_upload_key[model_file] = FileOps.download(
                    model_file, save_model)
            model_file = model_upload_key[model_file]

            task.model.model = save_model

            for _task in task.tasks:
                _task.model = FileOps.join_path(
                    self.edge_output_url, task_type, os.path.basename(model_file))
                sample_dir = FileOps.join_path(
                    self.edge_output_url, task_type,
                    f"{_task.samples.data_type}_{_task.entry}.sample")
                _task.samples.data_url = FileOps.download(
                    _task.samples.data_url, sample_dir)

        save_extractor = FileOps.join_path(
            self.edge_output_url, task_type,
            KBResourceConstant.TASK_EXTRACTOR_NAME.value
        )
        extractor = FileOps.dump(extractor, save_extractor)

        return extractor, task_groups

    def save_unseen_samples(self, samples, post_process):
        # TODO: save unseen samples to specified directory.
        if callable(post_process):
            samples = post_process(samples)

        fd, name = tempfile.mkstemp()

        FileOps.dump(samples, name)
        unseen_save_url = FileOps.join_path(
            Context.get_parameters(
                "unseen_save_url",
                self.edge_output_url),
            f"unseen_samples_{time.time()}.pkl")
        return FileOps.upload(name, unseen_save_url)


class ModelLoadingThread(threading.Thread):
    """Hot task index loading with multithread support"""
    MODEL_MANIPULATION_SEM = threading.Semaphore(1)

    def __init__(self,
                 edge_knowledge_management,
                 callback=None
                 ):
        model_check_time = int(Context.get_parameters(
            "MODEL_POLL_PERIOD_SECONDS", "30")
        )
        if model_check_time < 1:
            LOGGER.warning("Catch an abnormal value in "
                           "`MODEL_POLL_PERIOD_SECONDS`, fallback with 60")
            model_check_time = 60

        self.edge_knowledge_management = edge_knowledge_management
        self.check_time = model_check_time
        self.callback = callback
        self.version = str(time.time())
        super(ModelLoadingThread, self).__init__()

    def run(self):
        while True:
            time.sleep(self.check_time)
            latest_task_index = Context.get_parameters("MODEL_URLS", None)
            if not latest_task_index:
                continue
            latest_task_index = FileOps.load(latest_task_index)
            latest_version = str(latest_task_index.get("create_time"))

            if latest_version == self.version:
                continue
            self.version = latest_version
            with self.MODEL_MANIPULATION_SEM:
                try:
                    FileOps.dump(latest_task_index, self.task_index)
                    # TODO: update local kb with the latest index.pkl
                    self.edge_knowledge_management.update_kb(self.task_index)

                    status = K8sResourceKindStatus.COMPLETED.value
                except Exception as e:
                    status = K8sResourceKindStatus.FAILED.value
                if self.callback:
                    self.callback(
                        task_info=None, status=status, kind="deploy"
                    )

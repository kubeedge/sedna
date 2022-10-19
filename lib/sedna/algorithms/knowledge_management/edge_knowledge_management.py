import os
import time
import tempfile
import threading

from sedna.common.log import LOGGER
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.file_ops import FileOps
from sedna.common.constant import KBResourceConstant, K8sResourceKindStatus

from .base_knowledge_management import BaseKnowledgeManagement

__all__ = ('EdgeKnowledgeManagement', )

@ClassFactory.register(ClassType.KM)
class EdgeKnowledgeManagement(BaseKnowledgeManagement):
    """
    Manage inference, knowledge base update, etc., at the edge.

    Parameters:
        ----------
    config: Dict
        parameters to initialize an object
    estimator: Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    """

    def __init__(self, config, seen_estimator, unseen_estimator, **kwargs):
        super(EdgeKnowledgeManagement, self).__init__(config, seen_estimator, unseen_estimator)

        self.edge_output_url = Context.get_parameters("edge_output_url", KBResourceConstant.EDGE_KB_DIR.value)
        self.task_index = FileOps.join_path(self.edge_output_url, KBResourceConstant.KB_INDEX_NAME.value)
        self.local_unseen_save_url = FileOps.join_path(self.edge_output_url, "unseen_samples")
        if not os.path.exists(self.local_unseen_save_url):
            os.makedirs(self.local_unseen_save_url)
        UnseenSampleUploadThread(self).start()

    def update_kb(self, task_index):
        if isinstance(task_index, str):
            try:
                task_index = FileOps.load(task_index)
            except Exception as err:
                self.log.error(f"{err}")
                self.log.error(
                    "Load task index failed. KB deployment to the edge failed.")
                return None

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
            "created_time": task_index.get("created_time", str(time.time()))
        }

        fd, name = tempfile.mkstemp()
        FileOps.dump(task_info, name)
        return FileOps.upload(name, self.task_index)

    def save_task_index(self, task_index, task_type="seen_task"):
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
        if callable(post_process):
            # customized sample saving function
            post_process(samples.x, self.local_unseen_save_url)
            return

        for sample in samples.x:
            if isinstance(sample, dict):
                img = sample.get("image")
                image_name = "{}.png".format(str(time.time()))
                image_url = FileOps.join_path(self.local_unseen_save_url, image_name)
                img.save(image_url)
            else:
                image_name = os.path.basename(sample[0])
                image_url = FileOps.join_path(self.local_unseen_save_url, image_name)
                FileOps.upload(sample[0], image_url, clean=False)

        LOGGER.info(f"Unseen sample uploading completes.")

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
        self.version = None
        self.edge_knowledge_management = edge_knowledge_management
        self.check_time = model_check_time
        self.callback = callback
        task_index = edge_knowledge_management.task_index
        if FileOps.exists(task_index):
            self.version = FileOps.load(task_index).get("create_time")

        super(ModelLoadingThread, self).__init__()

    def run(self):
        while True:
            time.sleep(self.check_time)
            latest_task_index = Context.get_parameters("MODEL_URLS", None)
            if not self.version:
                continue
            if not latest_task_index:
                continue
            latest_task_index = FileOps.load(latest_task_index)
            latest_version = str(latest_task_index.get("create_time"))

            if latest_version == self.version:
                continue
            self.version = latest_version
            with self.MODEL_MANIPULATION_SEM:
                LOGGER.info(
                    f"Update model start with version {self.version}")
                try:
                    task_index_url = \
                        FileOps.dump(latest_task_index, self.edge_knowledge_management.task_index)
                    # TODO: update local kb with the latest index.pkl
                    self.edge_knowledge_management.update_kb(task_index_url)

                    status = K8sResourceKindStatus.COMPLETED.value
                    LOGGER.info(f"Update task index complete "
                                f"with version {self.version}")
                except Exception as e:
                    LOGGER.error(f"fail to update task index: {e}")
                    status = K8sResourceKindStatus.FAILED.value
                if self.callback:
                    self.callback(
                        task_info=None, status=status, kind="deploy"
                    )

class UnseenSampleUploadThread(threading.Thread):
    MODEL_MANIPULATION_SEM = threading.Semaphore(1)

    def __init__(self, edge_knowledge_management):
        self.local_unseen_save_url = edge_knowledge_management.local_unseen_save_url
        self.unseen_save_url = Context.get_parameters("unseen_save_url", "/tmp")
        self.check_time = 5
        super(UnseenSampleUploadThread, self).__init__()

    def run(self):
        while True:
            time.sleep(self.check_time)
            if not self.local_unseen_save_url:
                continue
            samples = os.listdir(self.local_unseen_save_url)
            if len(samples) == 0:
                continue
            for sample in samples:
                local_sample_url = FileOps.join_path(self.local_unseen_save_url, sample)
                dest_sample_url = FileOps.join_path(self.unseen_save_url, sample)
                FileOps.upload(local_sample_url, dest_sample_url)



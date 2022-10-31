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

import numpy as np

from sedna.core.base import JobBase
from sedna.common.file_ops import FileOps
from sedna.common.constant import K8sResourceKind, K8sResourceKindStatus
from sedna.common.constant import KBResourceConstant
from sedna.common.config import Context
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.seen_task_learning import SeenTaskLearning
from sedna.algorithms.unseen_task_processing import UnseenTaskProcessing
from sedna.service.client import KBClient
from sedna.algorithms.knowledge_management.cloud_knowledge_management \
    import CloudKnowledgeManagement
from sedna.algorithms.knowledge_management.edge_knowledge_management \
    import EdgeKnowledgeManagement


class LifelongLearning(JobBase):
    """
     Lifelong Learning (LL) is an advanced machine learning (ML) paradigm that
     learns continuously, accumulates the knowledge learned in the past, and
     uses/adapts it to help future learning and problem solving.

     Sedna provide the related interfaces for application development.

    Parameters
    ----------
    estimator : Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    task_definition : Dict
        Divide multiple tasks based on data,
        see `task_definition.task_definition` for more detail.
    task_relationship_discovery : Dict
        Discover relationships between all tasks, see
        `task_relationship_discovery.task_relationship_discovery` for more detail.
    task_allocation : Dict
        Mining seen tasks of inference sample,
        see `task_allocation.task_allocation` for more detail.
    task_remodeling : Dict
        Remodeling tasks based on their relationships,
        see `task_remodeling.task_remodeling` for more detail.
    inference_integrate : Dict
        Integrate the inference results of all related
        tasks, see `inference_integrate.inference_integrate` for more detail.
    task_update_decision: Dict
        Task update strategy making algorithms,
        see 'knowledge_management.task_update_decision.task_update_decision' for more detail.
    unseen_task_allocation: Dict
        Mining unseen tasks of inference sample,
        see `unseen_task_processing.unseen_task_allocation.unseen_task_allocation` for more detail.
    unseen_sample_recognition: Dict
        Dividing inference samples into seen tasks and unseen tasks,
        see 'unseen_task_processing.unseen_sample_recognition.unseen_sample_recognition' for more detail.
    unseen_sample_re_recognition: Dict
        Dividing unseen training samples into seen tasks and unseen tasks,
        see 'unseen_task_processing.unseen_sample_re_recognition.unseen_sample_re_recognition' for more detail.

    Examples
    --------
    >>> estimator = XGBClassifier(objective="binary:logistic")
    >>> task_definition = {
            "method": "TaskDefinitionByDataAttr",
            "param": {"attribute": ["season", "city"]}
        }
    >>> task_relationship_discovery = {
            "method": "DefaultTaskRelationDiscover", "param": {}
        }
    >>> task_mining = {
            "method": "TaskMiningByDataAttr",
            "param": {"attribute": ["season", "city"]}
        }
    >>> task_remodeling = None
    >>> inference_integrate = {
            "method": "DefaultInferenceIntegrate", "param": {}
        }
    >>> task_update_decision = {
            "method": "UpdateStrategyDefault", "param": {}
        }
    >>> unseen_task_allocation = {
            "method": "UnseenTaskAllocationDefault", "param": {}
        }
    >>> unseen_sample_recognition = {
            "method": "SampleRegonitionDefault", "param": {}
        }
    >>> unseen_sample_re_recognition = {
            "method": "SampleReRegonitionDefault", "param": {}
        }
    >>> ll_jobs = LifelongLearning(
            estimator,
            task_definition=None,
            task_relationship_discovery=None,
            task_allocation=None,
            task_remodeling=None,
            inference_integrate=None,
            task_update_decision=None,
            unseen_task_allocation=None,
            unseen_sample_recognition=None,
            unseen_sample_re_recognition=None,
        )
    """

    def __init__(self,
                 estimator,
                 task_definition=None,
                 task_relationship_discovery=None,
                 task_allocation=None,
                 task_remodeling=None,
                 inference_integrate=None,
                 task_update_decision=None,
                 unseen_task_allocation=None,
                 unseen_sample_recognition=None,
                 unseen_sample_re_recognition=None
                 ):

        e = SeenTaskLearning(
            estimator=estimator,
            task_definition=task_definition,
            task_relationship_discovery=task_relationship_discovery,
            seen_task_allocation=task_allocation,
            task_remodeling=task_remodeling,
            inference_integrate=inference_integrate
        )

        self.unseen_sample_recognition = unseen_sample_recognition or {
            "method": "SampleRegonitionDefault"
        }
        self.unseen_sample_recognition_param = e._parse_param(
            self.unseen_sample_recognition.get("param", {}))

        self.unseen_sample_re_recognition = unseen_sample_re_recognition or {
            "method": "SampleReRegonitionDefault"
        }
        self.unseen_sample_re_recognition_param = e._parse_param(
            self.unseen_sample_re_recognition.get("param", {}))

        self.task_update_decision = task_update_decision or {
            "method": "UpdateStrategyDefault"
        }
        self.task_update_decision_param = e._parse_param(
            self.task_update_decision.get("param", {})
        )

        config = dict(
            ll_kb_server=Context.get_parameters("KB_SERVER"),
            output_url=Context.get_parameters(
                "OUTPUT_URL",
                "/tmp"),
            cloud_output_url=Context.get_parameters(
                "OUTPUT_URL",
                "/tmp"),
            edge_output_url=Context.get_parameters(
                "EDGE_OUTPUT_URL",
                KBResourceConstant.EDGE_KB_DIR.value),
            task_index=KBResourceConstant.KB_INDEX_NAME.value)

        self.cloud_knowledge_management = CloudKnowledgeManagement(
            config, estimator=e)

        self.edge_knowledge_management = EdgeKnowledgeManagement(
            config, estimator=e)

        self.unseen_task_processing = UnseenTaskProcessing(
            estimator,
            config,
            self.cloud_knowledge_management,
            self.edge_knowledge_management,
            unseen_task_allocation)

        task_index = FileOps.join_path(config['output_url'],
                                       KBResourceConstant.KB_INDEX_NAME.value)

        config['task_index'] = task_index
        super(LifelongLearning, self).__init__(
            estimator=e, config=config
        )
        self.job_kind = K8sResourceKind.LIFELONG_JOB.value
        self.kb_server = KBClient(kbserver=self.config.ll_kb_server)

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              action="initial",
              **kwargs):
        """
        fit for update the knowledge based on training data.

        Parameters
        ----------
        train_data : BaseDataSource
            Train data, see `sedna.datasources.BaseDataSource` for more detail.
        valid_data : BaseDataSource
            Valid data, BaseDataSource or None.
        post_process : function
            function or a registered method, callback after `estimator` train.
        action : str
            `update` or `initial` the knowledge base
        kwargs : Dict
            parameters for `estimator` training, Like:
            `early_stopping_rounds` in Xgboost.XGBClassifier

        Returns
        -------
        train_history : object
        """

        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        res, seen_task_index = self.estimator.train(
            train_data=train_data,
            valid_data=valid_data,
            **kwargs
        )  # todo: Distinguishing incremental update and fully overwrite

        unseen_res, unseen_task_index = self.unseen_task_processing.initialize()

        task_index = dict(
            seen_task=seen_task_index,
            unseen_task=unseen_task_index)
        task_index_url = FileOps.dump(
            task_index, self.cloud_knowledge_management.local_task_index_url)

        task_index = self.cloud_knowledge_management.update_kb(
            task_index_url, self.kb_server)
        res.update(unseen_res)

        task_info_res = self.estimator.model_info(
            task_index,
            relpath=self.config.data_path_prefix)
        self.report_task_info(
            None, K8sResourceKindStatus.COMPLETED.value, task_info_res)
        self.log.info(f"Lifelong learning Train task Finished, "
                      f"KB index save in {task_index}")
        return callback_func(self.estimator, res) if callback_func else res

    def update(self, train_data, valid_data=None, post_process=None, **kwargs):
        """
        fit for update the knowledge based on incremental data.

        Parameters
        ----------
        train_data : BaseDataSource
            Train data, see `sedna.datasources.BaseDataSource` for more detail.
        valid_data : BaseDataSource
            Valid data, BaseDataSource or None.
        post_process : function
            function or a registered method, callback after `estimator` train.
        kwargs : Dict
            parameters for `estimator` training, Like:
            `early_stopping_rounds` in Xgboost.XGBClassifier

        Returns
        -------
        train_history : object
        """
        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        task_index_url = self.get_parameters(
            "CLOUD_KB_INDEX", self.cloud_knowledge_management.task_index)
        index_url = self.cloud_knowledge_management.local_task_index_url
        FileOps.download(task_index_url, index_url)

        unseen_sample_re_recognition = ClassFactory.get_cls(
            ClassType.UTD, self.unseen_sample_re_recognition["method"])(
            index_url, **self.unseen_sample_re_recognition_param)

        seen_samples, unseen_samples = unseen_sample_re_recognition(train_data)

        # TODO: retrain temporarily
        # historical_data = self._fetch_historical_data(index_url)
        # seen_samples.x = np.concatenate(
        #     (historical_data.x, seen_samples.x, unseen_samples.x), axis=0)
        # seen_samples.y = np.concatenate(
        #     (historical_data.y, seen_samples.y, unseen_samples.y), axis=0)

        seen_samples.x = np.concatenate(
            (seen_samples.x, unseen_samples.x), axis=0)
        seen_samples.y = np.concatenate(
            (seen_samples.y, unseen_samples.y), axis=0)

        task_update_decision = ClassFactory.get_cls(
            ClassType.KM, self.task_update_decision["method"])(
            index_url, **self.task_update_decision_param)

        tasks, task_update_strategies = task_update_decision(
            seen_samples, task_type="seen_task")
        seen_task_index = self.cloud_knowledge_management.estimator.update(
            tasks, task_update_strategies, task_index=index_url)

        tasks, task_update_strategies = task_update_decision(
            unseen_samples, task_type="unseen_task")
        unseen_task_index = self.unseen_task_processing.update(
            tasks, task_update_strategies, task_index=index_url)

        task_index = {
            "seen_task": seen_task_index,
            "unseen_task": unseen_task_index,
        }

        task_index = self.cloud_knowledge_management.update_kb(
            task_index, self.kb_server)

        task_info_res = self.estimator.model_info(
            task_index,
            relpath=self.config.data_path_prefix)

        self.report_task_info(
            None, K8sResourceKindStatus.COMPLETED.value, task_info_res)
        self.log.info(f"Lifelong learning Update task Finished, "
                      f"KB index save in {task_index}")
        return callback_func(self.estimator,
                             task_index) if callback_func else task_index

    def evaluate(self, data, post_process=None, **kwargs):
        """
        evaluated the performance of each task from training, filter tasks
        based on the defined rules.

        Parameters
        ----------
        data : BaseDataSource
            valid data, see `sedna.datasources.BaseDataSource` for more detail.
        kwargs: Dict
            parameters for `estimator` evaluate, Like:
            `ntree_limit` in Xgboost.XGBClassifier
        """

        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        task_index_url = Context.get_parameters(
            "MODEL_URLS", self.cloud_knowledge_management.task_index)
        index_url = self.cloud_knowledge_management.local_task_index_url
        self.log.info(
            f"Download kb index from {task_index_url} to {index_url}")
        FileOps.download(task_index_url, index_url)

        res, index_file = self._task_evaluation(
            data, task_index=index_url, **kwargs)
        self.log.info("Task evaluation finishes.")

        FileOps.upload(index_file, self.cloud_knowledge_management.task_index)
        self.log.info(
            f"upload kb index from {index_file} to {self.cloud_knowledge_management.task_index}")
        task_info_res = self.estimator.model_info(
            self.cloud_knowledge_management.task_index, result=res,
            relpath=self.config.data_path_prefix)
        self.report_task_info(
            None,
            K8sResourceKindStatus.COMPLETED.value,
            task_info_res,
            kind="eval")
        return callback_func(res) if callback_func else res

    def inference(self, data=None, post_process=None, **kwargs):
        """
        predict the result for input data based on training knowledge.

        Parameters
        ----------
        data : BaseDataSource
            inference sample, see `sedna.datasources.BaseDataSource` for
            more detail.
        post_process: function
            function or a registered method,  effected after `estimator`
            prediction, like: label transform.
        kwargs: Dict
            parameters for `estimator` predict, Like:
            `ntree_limit` in Xgboost.XGBClassifier

        Returns
        -------
        """
        seen_res, unseen_res = None, None
        task_index_url = Context.get_parameters(
            "MODEL_URLS", self.cloud_knowledge_management.task_index)
        index_url = self.edge_knowledge_management.task_index
        if not FileOps.exists(index_url):
            FileOps.download(task_index_url, index_url)
            self.log.info(
                f"Download kb index from {task_index_url} to {index_url}")

            self.edge_knowledge_management.update_kb(index_url)
            self.log.info(f"Tasks are deployed at the edge.")

        unseen_sample_recognition = ClassFactory.get_cls(
            ClassType.UTD,
            self.unseen_sample_recognition["method"])(
            self.edge_knowledge_management.task_index,
            **self.unseen_sample_recognition_param)

        seen_samples, unseen_samples = unseen_sample_recognition(
            data, **kwargs)
        if unseen_samples.x is not None and len(unseen_samples.x) > 0:
            self.edge_knowledge_management.log.info(
                f"Unseen task is detected.")
            unseen_res, unseen_tasks = self.unseen_task_processing.predict(
                unseen_samples)

            unseen_save_url = self.edge_knowledge_management.save_unseen_samples(
                unseen_samples, post_process=post_process)
            self.log.info(
                f"Unseen samples are being uploaded to {unseen_save_url}.")

        if seen_samples.x is not None and len(seen_samples.x) > 0:
            seen_res, seen_tasks = self.edge_knowledge_management.estimator.predict(
                data=seen_samples, post_process=post_process,
                task_index=index_url,
                task_type="seen_task",
                **kwargs
            )

        return seen_res, unseen_res

    def _task_evaluation(self, data, **kwargs):
        res, tasks_detail = self.cloud_knowledge_management.estimator.evaluate(
            data=data, **kwargs)
        drop_task = self.cloud_knowledge_management.evaluate_tasks(
            tasks_detail, **kwargs)

        index_file = self.kb_server.update_task_status(drop_task, new_status=0)

        if not index_file:
            self.log.error(f"KB update Fail !")
            index_file = str(
                self.cloud_knowledge_management.local_task_index_url)
        else:
            self.log.info(f"Deploy {index_file} to the edge.")

        return res, index_file

    def _fetch_historical_data(self, task_index):
        if isinstance(task_index, str):
            task_index = FileOps.load(task_index)

        samples = BaseDataSource(data_type="train")

        for task_group in task_index["seen_task"]["task_groups"]:
            if isinstance(task_group.samples, BaseDataSource):
                _samples = task_group.samples
            else:
                _samples = FileOps.load(task_group.samples.data_url)

            samples.x = _samples.x if samples.x is None else np.concatenate(
                (samples.x, _samples.x), axis=0)
            samples.y = _samples.y if samples.y is None else np.concatenate(
                (samples.y, _samples.y), axis=0)

        return samples

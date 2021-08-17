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
import tempfile

from sedna.backend import set_backend
from sedna.core.base import JobBase
from sedna.common.file_ops import FileOps
from sedna.common.constant import K8sResourceKind, K8sResourceKindStatus
from sedna.common.constant import KBResourceConstant
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.multi_task_learning import MulTaskLearning
from sedna.service.client import KBClient


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
        see `task_jobs.task_definition` for more detail.
    task_relationship_discovery : Dict
        Discover relationships between all tasks, see
        `task_jobs.task_relationship_discovery` for more detail.
    task_mining : Dict
        Mining tasks of inference sample,
        see `task_jobs.task_mining` for more detail.
    task_remodeling : Dict
        Remodeling tasks based on their relationships,
        see `task_jobs.task_remodeling` for more detail.
    inference_integrate : Dict
        Integrate the inference results of all related
        tasks, see `task_jobs.inference_integrate` for more detail.
    unseen_task_detect: Dict
        unseen task detect algorithms with parameters which has registered to
        ClassFactory, see `sedna.algorithms.unseen_task_detect` for more detail


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
    >>> unseen_task_detect = {
            "method": "TaskAttrFilter", "param": {}
        }
    >>> ll_jobs = LifelongLearning(
            estimator=estimator,
            task_definition=task_definition,
            task_relationship_discovery=task_relationship_discovery,
            task_mining=task_mining,
            task_remodeling=task_remodeling,
            inference_integrate=inference_integrate,
            unseen_task_detect=unseen_task_detect
        )
    """

    def __init__(self,
                 estimator,
                 task_definition=None,
                 task_relationship_discovery=None,
                 task_mining=None,
                 task_remodeling=None,
                 inference_integrate=None,
                 unseen_task_detect=None):

        if not task_definition:
            task_definition = {
                "method": "TaskDefinitionByDataAttr"
            }
        if not unseen_task_detect:
            unseen_task_detect = {
                "method": "TaskAttrFilter"
            }
        e = MulTaskLearning(
            estimator=estimator,
            task_definition=task_definition,
            task_relationship_discovery=task_relationship_discovery,
            task_mining=task_mining,
            task_remodeling=task_remodeling,
            inference_integrate=inference_integrate)
        self.unseen_task_detect = unseen_task_detect.get("method",
                                                         "TaskAttrFilter")
        self.unseen_task_detect_param = e._parse_param(
            unseen_task_detect.get("param", {})
        )
        config = dict(
            ll_kb_server=Context.get_parameters("KB_SERVER"),
            output_url=Context.get_parameters("OUTPUT_URL", "/tmp")
        )
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
        res, task_index_url = self.estimator.train(
            train_data=train_data,
            valid_data=valid_data,
            **kwargs
        )  # todo: Distinguishing incremental update and fully overwrite

        if isinstance(task_index_url, str) and FileOps.exists(task_index_url):
            task_index = FileOps.load(task_index_url)
        else:
            task_index = task_index_url

        extractor = task_index['extractor']
        task_groups = task_index['task_groups']

        model_upload_key = {}
        for task in task_groups:
            model_file = task.model.model
            save_model = FileOps.join_path(
                self.config.output_url,
                os.path.basename(model_file)
            )
            if model_file not in model_upload_key:
                model_upload_key[model_file] = FileOps.upload(model_file,
                                                              save_model)
            model_file = model_upload_key[model_file]

            try:
                model = self.kb_server.upload_file(save_model)
            except Exception as err:
                self.log.error(
                    f"Upload task model of {model_file} fail: {err}"
                )
                model = set_backend(
                    estimator=self.estimator.estimator.base_model
                )
                model.load(model_file)
            task.model.model = model

            for _task in task.tasks:
                sample_dir = FileOps.join_path(
                    self.config.output_url,
                    f"{_task.samples.data_type}_{_task.entry}.sample")
                task.samples.save(sample_dir)
                try:
                    sample_dir = self.kb_server.upload_file(sample_dir)
                except Exception as err:
                    self.log.error(
                        f"Upload task samples of {_task.entry} fail: {err}")
                _task.samples.data_url = sample_dir

        save_extractor = FileOps.join_path(
            self.config.output_url,
            KBResourceConstant.TASK_EXTRACTOR_NAME.value
        )
        extractor = FileOps.dump(extractor, save_extractor)
        try:
            extractor = self.kb_server.upload_file(extractor)
        except Exception as err:
            self.log.error(f"Upload task extractor fail: {err}")
        task_info = {
            "task_groups": task_groups,
            "extractor": extractor
        }
        fd, name = tempfile.mkstemp()
        FileOps.dump(task_info, name)

        index_file = self.kb_server.update_db(name)
        if not index_file:
            self.log.error(f"KB update Fail !")
            index_file = name
        FileOps.upload(index_file, self.config.task_index)

        task_info_res = self.estimator.model_info(
            self.config.task_index,
            relpath=self.config.data_path_prefix)
        self.report_task_info(
            None, K8sResourceKindStatus.COMPLETED.value, task_info_res)
        self.log.info(f"Lifelong learning Train task Finished, "
                      f"KB idnex save in {self.config.task_index}")
        return callback_func(self.estimator, res) if callback_func else res

    def update(self, train_data, valid_data=None, post_process=None, **kwargs):
        return self.train(
            train_data=train_data,
            valid_data=valid_data,
            post_process=post_process,
            action="update",
            **kwargs
        )

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
        task_index_url = self.get_parameters(
            "MODEL_URLS", self.config.task_index)
        index_url = self.estimator.estimator.task_index_url
        self.log.info(
            f"Download kb index from {task_index_url} to {index_url}")
        FileOps.download(task_index_url, index_url)
        res, tasks_detail = self.estimator.evaluate(data=data, **kwargs)
        drop_tasks = []

        model_filter_operator = self.get_parameters("operator", ">")
        model_threshold = float(self.get_parameters('model_threshold', 0.1))

        operator_map = {
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            "=": lambda x, y: x == y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }
        if model_filter_operator not in operator_map:
            self.log.warn(
                f"operator {model_filter_operator} use to "
                f"compare is not allow, set to <"
            )
            model_filter_operator = "<"
        operator_func = operator_map[model_filter_operator]

        for detail in tasks_detail:
            scores = detail.scores
            entry = detail.entry
            self.log.info(f"{entry} scores: {scores}")
            if any(map(lambda x: operator_func(float(x),
                                               model_threshold),
                       scores.values())):
                self.log.warn(
                    f"{entry} will not be deploy because all "
                    f"scores {model_filter_operator} {model_threshold}")
                drop_tasks.append(entry)
                continue
        drop_task = ",".join(drop_tasks)
        index_file = self.kb_server.update_task_status(drop_task, new_status=0)
        if not index_file:
            self.log.error(f"KB update Fail !")
            index_file = str(index_url)
        self.log.info(
            f"upload kb index from {index_file} to {self.config.task_index}")
        FileOps.upload(index_file, self.config.task_index)
        task_info_res = self.estimator.model_info(
            self.config.task_index, result=res,
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
        result : array_like
            results array, contain all inference results in each sample.
        is_unseen_task : bool
            `true` means detect an unseen task, `false` means not
        tasks : List
            tasks assigned to each sample.
        """
        task_index_url = self.get_parameters(
            "MODEL_URLS", self.config.task_index)
        index_url = self.estimator.estimator.task_index_url
        FileOps.download(task_index_url, index_url)
        res, tasks = self.estimator.predict(
            data=data, post_process=post_process, **kwargs
        )

        is_unseen_task = False
        if self.unseen_task_detect:

            try:
                if callable(self.unseen_task_detect):
                    unseen_task_detect_algorithm = self.unseen_task_detect()
                else:
                    unseen_task_detect_algorithm = ClassFactory.get_cls(
                        ClassType.UTD, self.unseen_task_detect
                    )()
            except ValueError as err:
                self.log.error("Lifelong learning "
                               "Inference [UTD] : {}".format(err))
            else:
                is_unseen_task = unseen_task_detect_algorithm(
                    tasks=tasks, result=res, **self.unseen_task_detect_param
                )
        return res, is_unseen_task, tasks

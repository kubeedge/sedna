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

"""Multiple task transfer learning algorithms"""

import json

from sedna.datasources import BaseDataSource
from sedna.backend import set_backend
from sedna.common.log import LOGGER
from sedna.common.file_ops import FileOps
from sedna.common.config import Context
from sedna.common.constant import KBResourceConstant
from sedna.common.class_factory import ClassFactory, ClassType

from .task_jobs.artifact import Model, Task, TaskGroup

__all__ = ('MulTaskLearning',)


class MulTaskLearning:
    """
    An auto machine learning framework for edge-cloud multitask learning

    See Also
    --------
    Train: Data + Estimator -> Task Definition -> Task Relationship Discovery
           -> Feature Engineering -> Training
    Inference: Data -> Task Allocation -> Task Mining -> Feature Engineering
               -> Task Remodeling -> Inference

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

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from sedna.algorithms.multi_task_learning import MulTaskLearning
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
    >>> mul_task_instance = MulTaskLearning(
            estimator=estimator,
            task_definition=task_definition,
            task_relationship_discovery=task_relationship_discovery,
            task_mining=task_mining,
            task_remodeling=task_remodeling,
            inference_integrate=inference_integrate
        )

    Notes
    -----
    All method defined under `task_jobs` and registered in `ClassFactory`.
    """

    _method_pair = {
        'TaskDefinitionBySVC': 'TaskMiningBySVC',
        'TaskDefinitionByDataAttr': 'TaskMiningByDataAttr',
    }

    def __init__(self,
                 estimator=None,
                 task_definition=None,
                 task_relationship_discovery=None,
                 task_mining=None,
                 task_remodeling=None,
                 inference_integrate=None
                 ):

        self.task_definition = task_definition or {
            "method": "TaskDefinitionByDataAttr"
        }
        self.task_relationship_discovery = task_relationship_discovery or {
            "method": "DefaultTaskRelationDiscover"
        }
        self.task_mining = task_mining or {}
        self.task_remodeling = task_remodeling or {
            "method": "DefaultTaskRemodeling"
        }
        self.inference_integrate = inference_integrate or {
            "method": "DefaultInferenceIntegrate"
        }
        self.models = None
        self.extractor = None
        self.base_model = estimator
        self.task_groups = None
        self.task_index_url = KBResourceConstant.KB_INDEX_NAME.value
        self.min_train_sample = int(Context.get_parameters(
            "MIN_TRAIN_SAMPLE", KBResourceConstant.MIN_TRAIN_SAMPLE.value
        ))

    @staticmethod
    def _parse_param(param_str):
        if not param_str:
            return {}
        if isinstance(param_str, dict):
            return param_str
        try:
            raw_dict = json.loads(param_str, encoding="utf-8")
        except json.JSONDecodeError:
            raw_dict = {}
        return raw_dict

    def _task_definition(self, samples):
        """
        Task attribute extractor and multi-task definition
        """
        method_name = self.task_definition.get(
            "method", "TaskDefinitionByDataAttr"
        )
        extend_param = self._parse_param(
            self.task_definition.get("param")
        )
        method_cls = ClassFactory.get_cls(
            ClassType.MTL, method_name)(**extend_param)
        return method_cls(samples)

    def _task_relationship_discovery(self, tasks):
        """
        Merge tasks from task_definition
        """
        method_name = self.task_relationship_discovery.get("method")
        extend_param = self._parse_param(
            self.task_relationship_discovery.get("param")
        )
        method_cls = ClassFactory.get_cls(
            ClassType.MTL, method_name)(**extend_param)
        return method_cls(tasks)

    def _task_mining(self, samples):
        """
        Mining tasks of inference sample base on task attribute extractor
        """
        method_name = self.task_mining.get("method")
        extend_param = self._parse_param(
            self.task_mining.get("param")
        )

        if not method_name:
            task_definition = self.task_definition.get(
                "method", "TaskDefinitionByDataAttr"
            )
            method_name = self._method_pair.get(task_definition,
                                                'TaskMiningByDataAttr')
            extend_param = self._parse_param(
                self.task_definition.get("param"))
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(
            task_extractor=self.extractor, **extend_param
        )
        return method_cls(samples=samples)

    def _task_remodeling(self, samples, mappings):
        """
        Remodeling tasks from task mining
        """
        method_name = self.task_remodeling.get("method")
        extend_param = self._parse_param(
            self.task_remodeling.get("param"))
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(
            models=self.models, **extend_param)
        return method_cls(samples=samples, mappings=mappings)

    def _inference_integrate(self, tasks):
        """
        Aggregate inference results from target models
        """
        method_name = self.inference_integrate.get("method")
        extend_param = self._parse_param(
            self.inference_integrate.get("param"))
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(
            models=self.models, **extend_param)
        return method_cls(tasks=tasks) if method_cls else tasks

    def train(self, train_data: BaseDataSource,
              valid_data: BaseDataSource = None,
              post_process=None, **kwargs):
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
        kwargs : Dict
            parameters for `estimator` training, Like:
            `early_stopping_rounds` in Xgboost.XGBClassifier

        Returns
        -------
        feedback : Dict
            contain all training result in each tasks.
        task_index_url : str
            task extractor model path, used for task mining.
        """

        tasks, task_extractor, train_data = self._task_definition(train_data)
        self.extractor = task_extractor
        task_groups = self._task_relationship_discovery(tasks)
        self.models = []
        callback = None
        if isinstance(post_process, str):
            callback = ClassFactory.get_cls(ClassType.CALLBACK, post_process)()
        self.task_groups = []
        feedback = {}
        rare_task = []
        for i, task in enumerate(task_groups):
            if not isinstance(task, TaskGroup):
                rare_task.append(i)
                self.models.append(None)
                self.task_groups.append(None)
                continue
            if not (task.samples and len(task.samples)
                    > self.min_train_sample):
                self.models.append(None)
                self.task_groups.append(None)
                rare_task.append(i)
                n = len(task.samples)
                LOGGER.info(f"Sample {n} of {task.entry} will be merge")
                continue
            LOGGER.info(f"MTL Train start {i} : {task.entry}")

            model = None
            for t in task.tasks:  # if model has train in tasks
                if not (t.model and t.result):
                    continue
                model_path = t.model.save(model_name=f"{task.entry}.model")
                t.model = model_path
                model = Model(index=i, entry=t.entry,
                              model=model_path, result=t.result)
                model.meta_attr = t.meta_attr
                break
            if not model:
                model_obj = set_backend(estimator=self.base_model)
                res = model_obj.train(train_data=task.samples, **kwargs)
                if callback:
                    res = callback(model_obj, res)
                model_path = model_obj.save(model_name=f"{task.entry}.model")
                model = Model(index=i, entry=task.entry,
                              model=model_path, result=res)

                model.meta_attr = [t.meta_attr for t in task.tasks]
            task.model = model
            self.models.append(model)
            feedback[task.entry] = model.result
            self.task_groups.append(task)

        if len(rare_task):
            model_obj = set_backend(estimator=self.base_model)
            res = model_obj.train(train_data=train_data, **kwargs)
            model_path = model_obj.save(model_name="global.model")
            for i in rare_task:
                task = task_groups[i]
                entry = getattr(task, 'entry', "global")
                if not isinstance(task, TaskGroup):
                    task = TaskGroup(
                        entry=entry, tasks=[]
                    )
                model = Model(index=i, entry=entry,
                              model=model_path, result=res)
                model.meta_attr = [t.meta_attr for t in task.tasks]
                task.model = model
                task.samples = train_data
                self.models[i] = model
                feedback[entry] = res
                self.task_groups[i] = task

        task_index = {
            "extractor": self.extractor,
            "task_groups": self.task_groups
        }
        if valid_data:
            feedback, _ = self.evaluate(valid_data, **kwargs)
        try:
            FileOps.dump(task_index, self.task_index_url)
        except TypeError:
            return feedback, task_index
        return feedback, self.task_index_url

    def load(self, task_index_url=None):
        """
        load task_detail (tasks/models etc ...) from task index file.
        It'll automatically loaded during `inference` and `evaluation` phases.

        Parameters
        ----------
        task_index_url : str
            task index file path, default self.task_index_url.
        """

        if task_index_url:
            self.task_index_url = task_index_url
        assert FileOps.exists(self.task_index_url), FileExistsError(
            f"Task index miss: {self.task_index_url}"
        )
        task_index = FileOps.load(self.task_index_url)
        self.extractor = task_index['extractor']
        if isinstance(self.extractor, str):
            self.extractor = FileOps.load(self.extractor)
        self.task_groups = task_index['task_groups']
        self.models = [task.model for task in self.task_groups]

    def predict(self, data: BaseDataSource,
                post_process=None, **kwargs):
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
        tasks : List
            tasks assigned to each sample.
        """

        if not (self.models and self.extractor):
            self.load()

        data, mappings = self._task_mining(samples=data)
        samples, models = self._task_remodeling(samples=data,
                                                mappings=mappings)

        callback = None
        if post_process:
            callback = ClassFactory.get_cls(ClassType.CALLBACK, post_process)()

        tasks = []
        for inx, df in enumerate(samples):
            m = models[inx]
            if not isinstance(m, Model):
                continue
            if isinstance(m.model, str):
                evaluator = set_backend(estimator=self.base_model)
                evaluator.load(m.model)
            else:
                evaluator = m.model
            pred = evaluator.predict(df.x, **kwargs)
            if callable(callback):
                pred = callback(pred, df)
            task = Task(entry=m.entry, samples=df)
            task.result = pred
            task.model = m
            tasks.append(task)
        res = self._inference_integrate(tasks)
        return res, tasks

    def evaluate(self, data: BaseDataSource,
                 metrics=None,
                 metrics_param=None,
                 **kwargs):
        """
        evaluated the performance of each task from training, filter tasks
        based on the defined rules.

        Parameters
        ----------
        data : BaseDataSource
            valid data, see `sedna.datasources.BaseDataSource` for more detail.
        metrics : function / str
            Metrics to assess performance on the task by given prediction.
        metrics_param : Dict
            parameter for metrics function.
        kwargs: Dict
            parameters for `estimator` evaluate, Like:
            `ntree_limit` in Xgboost.XGBClassifier

        Returns
        -------
        task_eval_res : Dict
            all metric results.
        tasks_detail : List[Object]
            all metric results in each task.
        """

        import pandas as pd
        from sklearn import metrics as sk_metrics

        result, tasks = self.predict(data, **kwargs)
        m_dict = {}
        if metrics:
            if callable(metrics):  # if metrics is a function
                m_name = getattr(metrics, '__name__', "mtl_eval")
                m_dict = {
                    m_name: metrics
                }
            elif isinstance(metrics, (set, list)):  # if metrics is multiple
                for inx, m in enumerate(metrics):
                    m_name = getattr(m, '__name__', f"mtl_eval_{inx}")
                    if isinstance(m, str):
                        m = getattr(sk_metrics, m)
                    if not callable(m):
                        continue
                    m_dict[m_name] = m
            elif isinstance(metrics, str):  # if metrics is single
                m_dict = {
                    metrics: getattr(sk_metrics, metrics, sk_metrics.log_loss)
                }
            elif isinstance(metrics, dict):  # if metrics with name
                for k, v in metrics.items():
                    if isinstance(v, str):
                        v = getattr(sk_metrics, v)
                    if not callable(v):
                        continue
                    m_dict[k] = v
        if not len(m_dict):
            m_dict = {
                'precision_score': sk_metrics.precision_score
            }
            metrics_param = {"average": "micro"}

        if isinstance(data.x, pd.DataFrame):
            data.x['pred_y'] = result
            data.x['real_y'] = data.y
        if not metrics_param:
            metrics_param = {}
        elif isinstance(metrics_param, str):
            metrics_param = self._parse_param(metrics_param)
        tasks_detail = []
        for task in tasks:
            sample = task.samples
            pred = task.result
            scores = {
                name: metric(sample.y, pred, **metrics_param)
                for name, metric in m_dict.items()
            }
            task.scores = scores
            tasks_detail.append(task)
        task_eval_res = {
            name: metric(data.y, result, **metrics_param)
            for name, metric in m_dict.items()
        }
        return task_eval_res, tasks_detail

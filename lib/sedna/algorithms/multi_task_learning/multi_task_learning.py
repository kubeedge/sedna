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
import json
import joblib

from sklearn import metrics as sk_metrics
from .task_jobs.artifact import Model, Task, TaskGroup
from sedna.datasources import BaseDataSource
from sedna.backend import set_backend
from sedna.common.log import sednaLogger
from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('MulTaskLearning',)


class MulTaskLearning:
    _method_pair = {
        'TaskDefinitionBySVC': 'TaskMiningBySVC',
        'TaskDefinitionByDataAttr': 'TaskMiningByDataAttr',
    }

    def __init__(self,
                 estimator=None,
                 task_definition="TaskDefinitionByDataAttr",
                 task_relationship_discovery=None,
                 task_mining=None,
                 task_remodeling=None,
                 inference_integrate=None,

                 task_definition_param=None,
                 task_relationship_discovery_param=None,
                 task_mining_param=None,
                 task_remodeling_param=None,
                 inference_integrate_param=None
                 ):
        self.method_selection = dict(
            task_definition=task_definition,
            task_relationship_discovery=task_relationship_discovery or "DefaultTaskRelationDiscover",
            task_mining=task_mining,
            task_remodeling=task_remodeling or "DefaultTaskRemodeling",
            inference_integrate=inference_integrate or "DefaultInferenceIntegrate",

            task_definition_param=task_definition_param,
            task_relationship_discovery_param=task_relationship_discovery_param,
            task_mining_param=task_mining_param,
            task_remodeling_param=task_remodeling_param,
            inference_integrate_param=inference_integrate_param
        )
        self.models = None
        self.extractor = None
        self.base_model = estimator
        self.task_groups = None
        self.task_index_url = Context.get_parameters(
            "MODEL_URLS", '/tmp/index.pkl'
        )

    @staticmethod
    def parse_param(param_str):
        if not param_str:
            return {}
        try:
            raw_dict = json.loads(param_str, encoding="utf-8")
        except json.JSONDecodeError:
            raw_dict = {}
        return raw_dict

    def task_definition(self, samples):
        method_name = self.method_selection.get("task_definition", "TaskDefinitionByDataAttr")
        extend_param = self.parse_param(self.method_selection.get("task_definition_param"))
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(**extend_param)
        return method_cls(samples)

    def task_relationship_discovery(self, tasks):
        method_name = self.method_selection.get("task_relationship_discovery")
        extend_param = self.parse_param(
            self.method_selection.get("task_relationship_discovery_param")
        )
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(**extend_param)
        return method_cls(tasks)

    def task_mining(self, samples):
        method_name = self.method_selection.get("task_mining")
        extend_param = self.parse_param(self.method_selection.get("task_mining_param"))

        if not method_name:
            task_definition = self.method_selection.get("task_definition",
                                                        "TaskDefinitionByDataAttr")
            method_name = self._method_pair.get(task_definition,
                                                'TaskMiningByDataAttr')
            extend_param = self.parse_param(self.method_selection.get("task_definition_param"))
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(
            task_extractor=self.extractor, **extend_param
        )
        return method_cls(samples=samples)

    def task_remodeling(self, samples, mappings):
        method_name = self.method_selection.get("task_remodeling")
        extend_param = self.parse_param(self.method_selection.get("task_remodeling_param"))
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(models=self.models, **extend_param)
        return method_cls(samples=samples, mappings=mappings)

    def inference_integrate(self, tasks):
        method_name = self.method_selection.get("inference_integrate")
        extend_param = self.parse_param(self.method_selection.get("inference_integrate_param"))
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(models=self.models, **extend_param)
        return method_cls(tasks=tasks) if method_cls else tasks

    def train(self, train_data: BaseDataSource,
              valid_data: BaseDataSource = None,
              post_process=None, **kwargs):
        tasks, task_extractor = self.task_definition(train_data)
        self.extractor = task_extractor
        self.task_groups = self.task_relationship_discovery(tasks)
        self.models = []
        callback = None
        if post_process:
            callback = ClassFactory.get_cls(ClassType.CALLBACK, post_process)()

        feedback = {}
        for i, task in enumerate(self.task_groups):
            if not isinstance(task, TaskGroup):
                continue
            sednaLogger.info(f"MTL Train start {i} : {task.entry}")
            model_obj = set_backend(estimator=self.base_model)
            res = model_obj.train(train_data=task.samples, **kwargs)
            if callback:
                res = callback(model_obj, res)
            model_path = model_obj.save(model_name=f"{task.entry}.pkl")
            model = Model(index=i, entry=task.entry,
                          model=model_path, result=res)
            model.meta_attr = [t.meta_attr for t in task.tasks]
            task.model = model
            self.models.append(model)
            feedback[task.entry] = res

        extractor_file = FileOps.join_path(
            os.path.dirname(self.task_index_url),
            "kb_extractor.pkl"
        )
        joblib.dump(self.extractor, extractor_file)
        task_index = {
            "extractor": extractor_file,
            "task_groups": self.task_groups
        }
        joblib.dump(task_index, self.task_index_url)
        if valid_data:
            feedback = self.evaluate(valid_data, **kwargs)

        return feedback

    def predict(self, data: BaseDataSource,
                post_process=None, **kwargs):
        if not (self.models and self.extractor):
            task_index = joblib.load(self.task_index_url)
            extractor_file = FileOps.join_path(
                os.path.dirname(self.task_index_url),
                "kb_extractor.pkl"
            )
            if not callable(task_index['extractor']) and \
                    isinstance(task_index['extractor'], str):
                FileOps.download(task_index['extractor'], extractor_file)
                self.extractor = joblib.load(extractor_file)
            else:
                self.extractor = task_index['extractor']
            self.task_groups = task_index['task_groups']
            self.models = [task.model for task in self.task_groups]
        data, mappings = self.task_mining(samples=data)
        samples, models = self.task_remodeling(samples=data, mappings=mappings)

        callback = None
        if post_process:
            callback = ClassFactory.get_cls(ClassType.CALLBACK, post_process)()

        tasks = []
        for inx, df in enumerate(samples):
            m = models[inx]
            if not isinstance(m, Model):
                continue
            model_obj = set_backend(estimator=self.base_model)
            evaluator = m.model if callable(m.model) else model_obj.load(m.model)
            pred = evaluator.predict(df.x, kwargs=kwargs)
            if callable(callback):
                pred = callback(pred, df)
            task = Task(entry=m.entry, samples=df)
            task.result = pred
            task.model = m
            tasks.append(task)
        res = self.inference_integrate(tasks)
        return res, tasks

    def evaluate(self, data: BaseDataSource,
                 metrics=None,
                 metrics_param=None,
                 **kwargs):
        result, tasks = self.predict(data, kwargs=kwargs)
        m_dict = {}
        if metrics:
            if callable(metrics):
                m_name = getattr(metrics, '__name__', "mtl_eval")
                m_dict = {
                    m_name: metrics
                }
            elif isinstance(metrics, (set, list)):
                for inx, m in enumerate(metrics):
                    m_name = getattr(m, '__name__', f"mtl_eval_{inx}")
                    if isinstance(m, str):
                        m = getattr(sk_metrics, m)
                    if not callable(m):
                        continue
                    m_dict[m_name] = m
            elif isinstance(metrics, str):
                m_dict = {
                    metrics: getattr(sk_metrics, metrics, sk_metrics.log_loss)
                }
            elif isinstance(metrics, dict):
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

        data.x['pred'] = result
        data.x['y'] = data.y
        if not metrics_param:
            metrics_param = {}
        elif isinstance(metrics_param, str):
            metrics_param = self.parse_param(metrics_param)
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
        return {
                   name: metric(data.y, result, **metrics_param)
                   for name, metric in m_dict.items()
               }, tasks_detail

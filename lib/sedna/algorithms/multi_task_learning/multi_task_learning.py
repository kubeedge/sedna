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
from copy import deepcopy
from sklearn import metrics as sk_metrics
from .task_jobs.artifact import Model, Task, TaskGroup
from sedna.datasources import BaseDataSource
from sedna.backend import set_backend
from sedna.common.log import sednaLogger
from sedna.common.config import Context
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('MulTaskLearning',)


class MulTaskLearning:
    _method_pair = {
        'TaskDefinitionBySVC': 'TaskMiningBySVC',
        'TaskDefinitionByDataAttr': 'TaskMiningByDataAttr',
    }

    def __init__(self, estimator=None, method_selection=None):
        self.method_selection = method_selection
        self.models = None
        self.extractor = None
        self.base_model = set_backend(estimator=estimator)
        self.task_groups = None
        self.task_index_url = Context.get_parameters(
            "KB_INDEX_URL", os.path.join(self.base_model.model_save_path, 'index.pkl')
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
        method_name = self.method_selection.get("task_relationship_discovery",
                                                "DefaultTaskRelationDiscover")
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
        method_name = self.method_selection.get("task_remodeling", "DefaultTaskRemodeling")
        extend_param = self.parse_param(self.method_selection.get("task_remodeling_param"))
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(models=self.models, **extend_param)
        return method_cls(samples=samples, mappings=mappings)

    def inference_integrate(self, tasks):
        method_name = self.method_selection.get("inference_integrate", "DefaultInferenceIntegrate")
        extend_param = self.parse_param(self.method_selection.get("inference_integrate_param"))
        method_cls = ClassFactory.get_cls(ClassType.MTL, method_name)(models=self.models, **extend_param)
        return method_cls(tasks=tasks) if method_cls else tasks

    def train(self, train_data: BaseDataSource,
              valid_data: BaseDataSource = None,
              post_process=None, **kwargs):
        tasks, task_extractor = self.task_definition(train_data)
        self.extractor = deepcopy(task_extractor)
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
            model_obj = deepcopy(self.base_model)
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

        task_index = {
            "extractor": task_extractor,
            "models": self.models
        }
        joblib.dump(task_index, self.task_index_url)
        if valid_data:
            feedback = self.evaluate(valid_data, kwargs=kwargs)

        return feedback

    def predict(self, data: BaseDataSource, post_process=None, **kwargs):
        if not (self.models and self.extractor):
            task_index = joblib.load(self.task_index_url)
            self.extractor = task_index['extractor']
            self.models = task_index['models']
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
            evaluator = self.base_model.load(m.model)
            pred = evaluator.predict(df.x, kwargs=kwargs)
            if callable(callback):
                pred = callback(pred, df)
            task = Task(entry=m.entry, samples=df)
            task.result = pred
            task.model = m
            tasks.append(task)
        res = self.inference_integrate(tasks)
        return res, tasks

    def evaluate(self, data: BaseDataSource, metrics="log_loss", metics_param=None, **kwargs):
        result = self.predict(data, kwargs=kwargs)
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
        data.x['pred'] = result
        data.x['y'] = data.y
        if not metics_param:
            metics_param = {}
        return {
            name: metric(data.y, result, **metics_param)
            for name, metric in m_dict.items()
        }

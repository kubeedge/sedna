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

from copy import deepcopy

from sedna.common.file_ops import FileOps
from sedna.common.constant import K8sResourceKind, K8sResourceKindStatus
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.core.base import JobBase

__all__ = ("IncrementalLearning",)


class IncrementalLearning(JobBase):
    """
    Incremental learning  is a method of machine learning in which input data
    is continuously used to extend the existing model's knowledge i.e. to
    further train the model. It represents a dynamic technique of supervised
    learning and unsupervised learning that can be applied when training data
    becomes available gradually over time.

    Sedna provide the related interfaces for application development.

    Parameters
    ----------
    estimator : Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    hard_example_mining : Dict
        HEM algorithms with parameters which has registered to ClassFactory,
        see `sedna.algorithms.hard_example_mining` for more detail.

    Examples
    --------
    >>> Estimator = keras.models.Sequential()
    >>> il_model = IncrementalLearning(
            estimator=Estimator,
            hard_example_mining={
                "method": "IBT",
                "param": {
                    "threshold_img": 0.9
                }
            }
        )

    Notes
    -----
    Sedna provide an interface call `get_hem_algorithm_from_config` to build
    the `hard_example_mining` parameter from CRD definition.
    """

    def __init__(self, estimator, hard_example_mining: dict = None):
        super(IncrementalLearning, self).__init__(estimator=estimator)

        self.model_urls = self.get_parameters(
            "MODEL_URLS")  # use in evaluation
        self.job_kind = K8sResourceKind.INCREMENTAL_JOB.value
        FileOps.clean_folder([self.config.model_url], clean=False)
        self.hard_example_mining_algorithm = None
        if not hard_example_mining:
            hard_example_mining = self.get_hem_algorithm_from_config()
        if hard_example_mining:
            hem = hard_example_mining.get("method", "IBT")
            hem_parameters = hard_example_mining.get("param", {})
            self.hard_example_mining_algorithm = ClassFactory.get_cls(
                ClassType.HEM, hem
            )(**hem_parameters)

    @classmethod
    def get_hem_algorithm_from_config(cls, **param):
        """
        get the `algorithm` name and `param` of hard_example_mining from crd

        Parameters
        ----------
        param : Dict
            update value in parameters of hard_example_mining

        Returns
        -------
        dict
            e.g.: {"method": "IBT", "param": {"threshold_img": 0.5}}

        Examples
        --------
        >>> IncrementalLearning.get_hem_algorithm_from_config(
                threshold_img=0.9
            )
        {"method": "IBT", "param": {"threshold_img": 0.9}}
        """
        return cls.parameters.get_algorithm_from_api(
            algorithm="HEM",
            **param
        )

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        """
        Training task for IncrementalLearning

        Parameters
        ----------
        train_data: BaseDataSource
            datasource use for train, see
            `sedna.datasources.BaseDataSource` for more detail.
        valid_data:  BaseDataSource
            datasource use for evaluation, see
            `sedna.datasources.BaseDataSource` for more detail.
        post_process: function or a registered method
            effected after `estimator` training.
        kwargs: Dict
            parameters for `estimator` training,
            Like:  `early_stopping_rounds` in Xgboost.XGBClassifier

        Returns
        -------
        estimator
        """

        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        res = self.estimator.train(
            train_data=train_data, valid_data=valid_data, **kwargs)
        model_paths = self.estimator.save(self.model_path)
        task_info_res = self.estimator.model_info(
            model_paths, result=res, relpath=self.config.data_path_prefix)
        self.report_task_info(
            None, K8sResourceKindStatus.COMPLETED.value, task_info_res)
        return callback_func(
            self.estimator) if callback_func else self.estimator

    def inference(self, data=None, post_process=None, **kwargs):
        """
        Inference task for IncrementalLearning

        Parameters
        ----------
        data: BaseDataSource
            datasource use for inference, see
            `sedna.datasources.BaseDataSource` for more detail.
        post_process: function or a registered method
            effected after `estimator` inference.
        kwargs: Dict
            parameters for `estimator` inference,
            Like:  `ntree_limit` in Xgboost.XGBClassifier

        Returns
        -------
        inference result : object
        result after post_process : object
        if is hard sample : bool
        """

        if not self.estimator.has_load:
            self.estimator.load(self.model_path)

        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        infer_res = self.estimator.predict(data, **kwargs)
        if callback_func:
            res = callback_func(
                deepcopy(infer_res)  # Prevent infer_result from being modified
            )
        else:
            res = infer_res
        is_hard_example = False

        if self.hard_example_mining_algorithm:
            is_hard_example = self.hard_example_mining_algorithm(res)
        return infer_res, res, is_hard_example

    def evaluate(self, data, post_process=None, **kwargs):
        """
        Evaluate task for IncrementalLearning

        Parameters
        ----------
        data: BaseDataSource
            datasource use for evaluation, see
            `sedna.datasources.BaseDataSource` for more detail.
        post_process: function or a registered method
            effected after `estimator` evaluation.
        kwargs: Dict
            parameters for `estimator` evaluate,
            Like:  `metric_name` in Xgboost.XGBClassifier

        Returns
        -------
        evaluate metrics : List
        """

        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        final_res = []
        all_models = []
        if self.model_urls:
            all_models = self.model_urls.split(";")
        elif self.config.model_url:
            all_models.append(self.config.model_url)
        for model_url in all_models:
            if not model_url.strip():
                continue
            self.estimator.model_save_path = model_url
            res = self.estimator.evaluate(
                data=data, model_path=model_url, **kwargs)
            if callback_func:
                res = callback_func(res)
            self.log.info(f"Evaluation with {model_url} : {res} ")
            task_info_res = self.estimator.model_info(
                model_url, result=res,
                relpath=self.config.data_path_prefix)
            if isinstance(
                    task_info_res, (list, tuple)
            ) and len(task_info_res):
                task_info_res = list(task_info_res)[0]
            final_res.append(task_info_res)
        self.report_task_info(None, K8sResourceKindStatus.COMPLETED.value,
                              final_res, kind="eval")

        return final_res

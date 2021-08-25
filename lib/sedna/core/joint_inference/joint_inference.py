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
from copy import deepcopy

from sedna.common.utils import get_host_ip
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.service.server import InferenceServer
from sedna.service.client import ModelClient, LCReporter
from sedna.common.constant import K8sResourceKind
from sedna.core.base import JobBase

__all__ = ("JointInference", "BigModelService")


class BigModelService(JobBase):
    """
    Large model services implemented
    Provides RESTful interfaces for large-model inference.

    Parameters
    ----------
    estimator : Instance, big model
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.

    Examples
    --------
    >>> Estimator = xgboost.XGBClassifier()
    >>> BigModelService(estimator=Estimator).start()
    """

    def __init__(self, estimator=None):
        """
        Initial a big model service for JointInference
        :param estimator: Customize estimator
        """

        super(BigModelService, self).__init__(estimator=estimator)
        self.local_ip = self.get_parameters("BIG_MODEL_BIND_IP", get_host_ip())
        self.port = int(self.get_parameters("BIG_MODEL_BIND_PORT", "5000"))

    def start(self):
        """
        Start inference rest server
        """

        if callable(self.estimator):
            self.estimator = self.estimator()
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"{self.model_path} miss")
        else:
            self.estimator.load(self.model_path)
        app_server = InferenceServer(model=self, servername=self.job_name,
                                     host=self.local_ip, http_port=self.port)
        app_server.start()

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        """todo: no support yet"""

    def inference(self, data=None, post_process=None, **kwargs):
        """
        Inference task for JointInference

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
        inference result
        """

        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        res = self.estimator.predict(data, **kwargs)
        if callback_func:
            res = callback_func(res)
        return res


class JointInference(JobBase):
    """
    Sedna provide a framework make sure under the condition of limited
    resources on the edge, difficult inference tasks are offloaded to the
    cloud to improve the overall performance, keeping the throughput.

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
    >>> ji_service = JointInference(
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

    def __init__(self, estimator=None, hard_example_mining: dict = None):
        super(JointInference, self).__init__(estimator=estimator)
        self.job_kind = K8sResourceKind.JOINT_INFERENCE_SERVICE.value
        self.local_ip = get_host_ip()
        self.remote_ip = self.get_parameters(
            "BIG_MODEL_IP", self.local_ip)
        self.port = int(self.get_parameters("BIG_MODEL_PORT", "5000"))

        report_msg = {
            "name": self.worker_name,
            "namespace": self.config.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": "inference",
            "results": []
        }
        period_interval = int(self.get_parameters("LC_PERIOD", "30"))
        self.lc_reporter = LCReporter(lc_server=self.config.lc_server,
                                      message=report_msg,
                                      period_interval=period_interval)
        self.lc_reporter.setDaemon(True)
        self.lc_reporter.start()

        if callable(self.estimator):
            self.estimator = self.estimator()
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"{self.model_path} miss")
        else:
            self.estimator.load(self.model_path)
        self.cloud = ModelClient(service_name=self.job_name,
                                 host=self.remote_ip, port=self.port)
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
        >>> JointInference.get_hem_algorithm_from_config(
                threshold_img=0.9
            )
        {"method": "IBT", "param": {"threshold_img": 0.9}}
        """
        return cls.parameters.get_algorithm_from_api(
            algorithm="HEM",
            **param
        )

    def inference(self, data=None, post_process=None, **kwargs):
        """
        Inference task with JointInference

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
        if is hard sample : bool
        inference result : object
        result from little-model : object
        result from big-model: object
        """

        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        res = self.estimator.predict(data, **kwargs)
        edge_result = deepcopy(res)

        if callback_func:
            res = callback_func(res)

        self.lc_reporter.update_for_edge_inference()

        is_hard_example = False
        cloud_result = None

        if self.hard_example_mining_algorithm:
            is_hard_example = self.hard_example_mining_algorithm(res)
            if is_hard_example:
                try:
                    cloud_data = self.cloud.inference(
                        data.tolist(), post_process=post_process, **kwargs)
                    cloud_result = cloud_data["result"]
                except Exception as err:
                    self.log.error(f"get cloud result error: {err}")
                else:
                    res = cloud_result
                self.lc_reporter.update_for_collaboration_inference()
        return [is_hard_example, res, edge_result, cloud_result]

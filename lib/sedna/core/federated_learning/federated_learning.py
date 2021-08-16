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


"""
Federated learning enables multiple actors to build a common, robust machine
learning model without sharing data, thus allowing to address critical issues
such as data privacy, data security, data access rights and access to
heterogeneous data.

Sedna provide the related interfaces for application development.

See `FederatedLearning` below.

"""

import time

from sedna.core.base import JobBase
from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.service.client import AggregationClient
from sedna.common.constant import K8sResourceKindStatus


class FederatedLearning(JobBase):
    def __init__(self, estimator, aggregation="FedAvg"):
        """
        Initial a FederatedLearning job

        Parameters
        ----------
        estimator: An instance with the high-level API that greatly simplifies
            machine learning programming. Estimators encapsulate training,
            evaluation, prediction, and exporting for your model.
        aggregation: aggregation algo which has registered to ClassFactory,
            see `sedna.algorithms.aggregation` for more detail.

        Examples
        --------
        >>> Estimator = keras.models.Sequential()
        >>> fl_model = FederatedLearning(
                estimator=Estimator,
                aggregation="FedAvg"
            )
        """

        protocol = Context.get_parameters("AGG_PROTOCOL", "ws")
        agg_ip = Context.get_parameters("AGG_IP", "127.0.0.1")
        agg_port = int(Context.get_parameters("AGG_PORT", "7363"))
        agg_uri = f"{protocol}://{agg_ip}:{agg_port}/{aggregation}"
        config = dict(
            protocol=protocol,
            agg_ip=agg_ip,
            agg_port=agg_port,
            agg_uri=agg_uri
        )
        super(FederatedLearning, self).__init__(
            estimator=estimator, config=config)
        self.aggregation = ClassFactory.get_cls(ClassType.FL_AGG, aggregation)

        connect_timeout = int(Context.get_parameters("CONNECT_TIMEOUT", "300"))
        self.node = None
        self.register(timeout=connect_timeout)

    def register(self, timeout=300):
        """
        Deprecated, Client proactively subscribes to the aggregation service.

        Parameters
        ----------
        timeout: int, connect timeout. Default: 300
        """
        self.log.info(
            f"Node {self.worker_name} connect to : {self.config.agg_uri}")
        self.node = AggregationClient(
            url=self.config.agg_uri,
            client_id=self.worker_name,
            ping_timeout=timeout
        )

        FileOps.clean_folder([self.config.model_url], clean=False)
        self.aggregation = self.aggregation()
        self.log.info(f"{self.worker_name} model prepared")
        if callable(self.estimator):
            self.estimator = self.estimator()

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        """
        Training task for FederatedLearning

        Parameters
        ----------
        train_data: datasource use for train, see
            `sedna.datasources.BaseDataSource` for more detail.
        valid_data: datasource use for evaluation, see
            `sedna.datasources.BaseDataSource` for more detail.
        post_process: function or a registered method,
            effected after `estimator` training.
        kwargs: parameters for `estimator` training,
            Like:  `early_stopping_rounds` in Xgboost.XGBClassifier
        """

        callback_func = None
        if post_process:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        round_number = 0
        num_samples = len(train_data)
        _flag = True
        start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        res = None
        while 1:
            if _flag:
                round_number += 1
                start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.log.info(
                    f"Federated learning start, round_number={round_number}")
                res = self.estimator.train(
                    train_data=train_data, valid_data=valid_data, **kwargs)

                current_weights = self.estimator.get_weights()
                send_data = {"num_samples": num_samples,
                             "weights": current_weights}
                self.node.send(
                    send_data, msg_type="update_weight", job_name=self.job_name
                )
            received = self.node.recv(wait_data_type="recv_weight")
            if not received:
                _flag = False
                continue
            _flag = True

            rec_data = received.get("data", {})
            exit_flag = rec_data.get("exit_flag", "")
            server_round = int(rec_data.get("round_number"))
            total_size = int(rec_data.get("total_sample"))
            self.log.info(
                f"Federated learning recv weight, "
                f"round: {server_round}, total_sample: {total_size}"
            )
            n_weight = rec_data.get("weights")
            self.estimator.set_weights(n_weight)
            task_info = {
                'currentRound': round_number,
                'sampleCount': total_size,
                'startTime': start,
                'updateTime': time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime())
            }
            model_paths = self.estimator.save()
            task_info_res = self.estimator.model_info(
                model_paths, result=res, relpath=self.config.data_path_prefix)
            if exit_flag == "ok":
                self.report_task_info(
                    task_info,
                    K8sResourceKindStatus.COMPLETED.value,
                    task_info_res)
                self.log.info(f"exit training from [{self.worker_name}]")
                return callback_func(
                    self.estimator) if callback_func else self.estimator
            else:
                self.report_task_info(
                    task_info,
                    K8sResourceKindStatus.RUNNING.value,
                    task_info_res)

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


import asyncio
import sys
import time

from sedna.algorithms.transmitter import S3Transmitter, WSTransmitter
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.config import BaseConfig, Context
from sedna.common.constant import K8sResourceKindStatus
from sedna.common.file_ops import FileOps
from sedna.core.base import JobBase
from sedna.service.client import AggregationClient

__all__ = ('FederatedLearning', 'FederatedLearningV2')


class FederatedLearning(JobBase):
    """
    Federated learning enables multiple actors to build a common, robust
    machine learning model without sharing data, thus allowing to address
    critical issues such as data privacy, data security, data access rights
    and access to heterogeneous data.

    Sedna provide the related interfaces for application development.

    Parameters
    ----------
    estimator: Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    aggregation: str
        aggregation algo which has registered to ClassFactory,
        see `sedna.algorithms.aggregation` for more detail.

    Examples
    --------
    >>> Estimator = keras.models.Sequential()
    >>> fl_model = FederatedLearning(
            estimator=Estimator,
            aggregation="FedAvg"
        )
    """

    def __init__(self, estimator, aggregation="FedAvg"):

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


class FederatedLearningV2:
    def __init__(self, data=None, estimator=None,
                 aggregation=None, transmitter=None) -> None:

        from plato.config import Config
        from plato.datasources import base
        # set parameters
        server = Config().server._asdict()
        clients = Config().clients._asdict()
        datastore = Config().data._asdict()
        train = Config().trainer._asdict()
        self.datasource = None
        if data is not None:
            if hasattr(data, "customized"):
                if data.customized:
                    self.datasource = base.DataSource()
                    self.datasource.trainset = data.trainset
                    self.datasource.testset = data.testset
            else:
                datastore.update(data.parameters)
                Config().data = Config.namedtuple_from_dict(datastore)

        self.model = None
        if estimator is not None:
            self.model = estimator.model
            train.update(estimator.hyperparameters)
            Config().trainer = Config.namedtuple_from_dict(train)

        if aggregation is not None:
            Config().algorithm = Config.namedtuple_from_dict(
                aggregation.parameters)
            if aggregation.parameters["type"] == "mistnet":
                clients["type"] = "mistnet"
                server["type"] = "mistnet"
            else:
                clients["do_test"] = True

        server["address"] = Context.get_parameters("AGG_IP")
        server["port"] = Context.get_parameters("AGG_PORT")

        if transmitter is not None:
            server.update(transmitter.parameters)

        Config().server = Config.namedtuple_from_dict(server)
        Config().clients = Config.namedtuple_from_dict(clients)

        from plato.clients import registry as client_registry
        self.client = client_registry.get(model=self.model,
                                          datasource=self.datasource)
        self.client.configure()

    @classmethod
    def get_transmitter_from_config(cls):
        if BaseConfig.transmitter == "ws":
            return WSTransmitter()
        elif BaseConfig.transmitter == "s3":
            return S3Transmitter(s3_endpoint_url=BaseConfig.s3_endpoint_url,
                                 access_key=BaseConfig.access_key_id,
                                 secret_key=BaseConfig.secret_access_key,
                                 transmitter_url=BaseConfig.agg_data_path)

    def train(self):
        if int(sys.version[2]) <= 6:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.client.start_client())
        else:
            asyncio.run(self.client.start_client())

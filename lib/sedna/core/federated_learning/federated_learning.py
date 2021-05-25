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
import time
import asyncio
from sedna.core.base import JobBase
from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.service.client import AggregationClient


class FederatedLearning(JobBase):
    """
    Federated learning Experiment
    """

    def __init__(self, estimator, aggregation="fedavg"):
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
        super(FederatedLearning, self).__init__(estimator=estimator, config=config)
        self.aggregation = ClassFactory.get_cls(ClassType.FLAGG, aggregation)
        self.node = None

    def connect(self):
        self.node = AggregationClient(url=self.config.agg_uri, client_id=self.worker_name)
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self.node.connect())
        self.log.info(f"Federated learning Experiment node register: {res}")
        FileOps.clean_folder([self.config.model_url], clean=False)
        self.aggregation = self.aggregation()
        self.log.info("Federated learning Experiment model prepared")
        ig callable(self.estimator):
            self.estimator = self.estimator()

    def train(self, train_data,
              valid_data=None,
              post_process=None,
              **kwargs):
        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(ClassType.CALLBACK, post_process)

        round_number = 0
        num_samples = len(train_data)
        self.aggregation.total_size += num_samples

        while 1:
            round_number += 1
            start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.log.info(f"Federated learning start at {start}, round_number={round_number}")
            res = self.estimator.train(train_data=train_data, valid_data=valid_data, **kwargs)

            self.aggregation.weights = self.estimator.get_weights()
            send_data = {"num_samples": num_samples, "weights": self.aggregation.weights}
            received = self.node.send(send_data, msg_type="update_weight", job_name=self.job_name)
            exit_flag = False
            if received and received["type"] == "update_weight" \
                    and received["job_name"] == self.job_name:
                recv = received["data"]

                rec_client = received["client"]
                rec_sample = recv["num_samples"]

                self.log.info(f"Federated learning get weight from [{rec_client}] : {rec_sample}")
                n_weight = self.aggregation.aggregation(recv["weights"], rec_sample)
                self.estimator.set_weights(n_weight)
                exit_flag = recv.get("exit_flag", "") == "ok"
            task_info = {
                'currentRound': round_number,
                'sampleCount': self.aggregation.total_size,
                'startTime': start,
                'updateTime': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            if exit_flag:
                self._report_task_info(task_info, 'completed', res)
                self.log.info(f"exit training from [{self.worker_name}]")
                self.estimator.save()

                return callback_func(self.estimator) if callback_func else self.estimator
            else:
                self._report_task_info(task_info, 'training', res)

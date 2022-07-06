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

from abc import ABC, abstractmethod
from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka import KafkaAdminClient
from sedna.common.log import LOGGER

from kafka.admin import NewTopic
from kafka.errors import NoBrokersAvailable


class Client(ABC):
    def __init__(self, address=["localhost"], port=[9092]) -> None:
        self.kafka_address = address
        self.kafka_port = port
        self.kafka_endpoints = []

        for addr, port in zip(self.kafka_address, self.kafka_port):
            LOGGER.debug(f"Adding address {addr} with \
                port {port} to list of endpoints.")
            self.kafka_endpoints.append(f"{addr}:{port}")

    @abstractmethod
    def connect(self, bootstrap_servers):
        raise NoBrokersAvailable

    # Call this function to connect to the brokers
    # when you have issues using the standard connect.
    def hardened_connect(self):
        try:
            self.connect(self.kafka_endpoints)
        except NoBrokersAvailable:
            LOGGER.info(
                "Switching to hardened connection procedure.")
            for srv in self.kafka_endpoints:
                try:
                    return self.connect(srv)
                except NoBrokersAvailable:
                    LOGGER.error(
                        f"Broker {srv} is not reachable, skipping")
                    continue

            raise NoBrokersAvailable


class AdminClient(Client):
    def __init__(self, address=["localhost"], port=[9092]) -> None:
        super().__init__(address, port)
        LOGGER.debug("Creating Kafka admin client")
        self.admin_client = KafkaAdminClient(
            bootstrap_servers=self.kafka_endpoints, request_timeout_ms=20000)

    def _parse_topics(self, topics, num_partitions, replication_factor):
        topic_list = []
        for topic in topics:
            topic_list.append(
                NewTopic(
                    name=topic,
                    num_partitions=num_partitions,
                    replication_factor=replication_factor))

        return topic_list

    def create_topics(self, topics, num_partitions=1, replication_factor=1):
        topic_list = self._parse_topics(
            topics, num_partitions, replication_factor)
        res = self.admin_client.create_topics(
            new_topics=topic_list, validate_only=False)
        return res

    def delete_topics(self, topics, num_partitions=1, replication_factor=1):
        topic_list = self._parse_topics(
            topics, num_partitions, replication_factor)
        res = self.admin_client.delete_topics(
            new_topics=topic_list, validate_only=False)
        return res

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

import pickle

from kafka.errors import KafkaError
from sedna.datasources.kafka import LOGGER, KafkaProducer, Client


class Producer(Client):
    def __init__(self, address=["localhost"], port=[9092]) -> None:
        super().__init__(address, port)

        LOGGER.debug("Creating Kafka producer")
        self.hardened_connect()

    def connect(self, boostrap_servers):
        self.producer = KafkaProducer(
            value_serializer=lambda v: pickle.dumps(
                v, protocol=pickle.HIGHEST_PROTOCOL),
            bootstrap_servers=boostrap_servers,
            max_request_size=10000000,
            linger_ms=1000
            )

    def publish_data_synchronous(self, data, topic="default"):
        # Publishes a message synchronously
        for t in topic:
            try:
                future = self.producer.send(t, data)
                record_metadata = future.get(timeout=3)
                LOGGER.info(
                    f"Message published on topic {record_metadata.topic} \
                         with offset {record_metadata.offset}.")
            except KafkaError as ke:
                LOGGER.error(f"Unable to write to Kafka. {ke}")
            except Exception as e:
                LOGGER.error(f"Something went wrong: {e}")
        return

    def publish_data_asynchronous(self, data, topic="default"):
        # Publishes a message
        for t in topic:
            try:
                self.producer.send(t, data).add_callback(
                    self.on_send_success).add_errback(self.on_send_error)
                LOGGER.debug(f"Message published on topic {topic}.")
            except Exception as e:
                LOGGER.error(f"Something went wrong: {e}")
        return

    def on_send_success(self, record_metadata):
        LOGGER.debug(f"Topic:{record_metadata.topic}")
        LOGGER.debug(f"Partition:{record_metadata.partition}")
        LOGGER.debug(f"Offset:{record_metadata.offset}")

    def on_send_error(self, excp):
        LOGGER.error('Unable to publish to Kafka', exc_info=excp)
        return

    def close(self):
        LOGGER.debug("Shutting down producer")
        self.producer.close()

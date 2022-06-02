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
from sedna.datasources.kafka import KafkaConsumer, LOGGER, Client


class Consumer(Client):
    def __init__(
        self,
        address=["localhost"], port=[9092],
        group_id="default",
        consumer_timeout_ms=250
    ) -> None:

        super().__init__(address, port)

        self.group_id = group_id
        self.consumer_timeout_ms = consumer_timeout_ms
        self.disconnected = False

        LOGGER.debug("Creating Kafka consumer")
        self.hardened_connect()

    def connect(self, boostrap_servers):
        self.consumer = KafkaConsumer(
                value_deserializer=lambda v: pickle.loads(v),
                group_id=self.group_id,
                bootstrap_servers=boostrap_servers,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                max_poll_interval_ms=10000,
                consumer_timeout_ms=self.consumer_timeout_ms
                )

    def get_topics(self):
        return self.consumer.topics()

    def subscribe(self, topic):
        for t in topic:
            try:
                LOGGER.debug(f"Subscribing to topics {t}.")
                self.consumer.subscribe(t)
            except Exception as e:
                LOGGER.error(
                    f"Unable to subscribe to topic {topic}. [{e}]")

    def consume_messages(self):
        try:
            LOGGER.debug("Reading messages")
            return list(map(lambda message: message.value, self.consumer))

        except Exception as e:
            LOGGER.error(
                f"Error while reading messages from Kafka broker:  [{e}]")
            return []

    def consume_messages_poll(self):
        data = []
        try:
            LOGGER.debug("Reading messages using poll")
            messages = self.consumer.poll(timeout_ms=1000)
            for key, record in messages.items():
                for item in record:
                    data.append(item.value)

            return data

        except Exception as e:
            LOGGER.error(
                f"Error while polling messages from Kafka broker: [{e}]")
            return []

    def pause(self):
        pass

    def resume(self):
        pass

    def close(self):
        LOGGER.debug("Shutting down consumer")
        self.disconnected = True
        self.consumer.close()

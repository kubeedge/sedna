import pickle

from sedna.datasources.kafka import *
from sedna.datasources.kafka.fluentd_reporter import FluentdReporter

POLL_TIMEOUT_MS = 5000

class Consumer(Client):
    def __init__(self, address = ["localhost"], port = [9092], group_id="default", _metric_reporters=[FluentdReporter]) -> None:
        super().__init__(address, port)
        LOGGER.debug("Creating Kafka consumer")
        self.consumer = KafkaConsumer(
            group_id=group_id,
            bootstrap_servers=self.kafka_endpoints,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            metric_reporters=_metric_reporters
            )

    def get_topics(self):
        return self.consumer.topics()

    def subscribe(self, topic):
        for t in topic:
            try:
                LOGGER.debug(f"Subscribing to topics {t}")
                self.consumer.subscribe(t)
            except Exception as e:
                LOGGER.error(f"Unable to subscribe to topic {e}")


    # This should work with callbacks?
    def consume_json_data(self):
        data = []

        if len(self.consumer.subscription()) == 0:
            LOGGER.error("Not subscribed to any topic")
            return data

        try:
            LOGGER.debug("Polling topics ...")
            records = self.consumer.poll(timeout_ms=POLL_TIMEOUT_MS)

            if len(records)==0:
                LOGGER.debug("No message(s) consumed (maybe we timed out waiting?)\n")

            LOGGER.debug("Consuming messages")
            for key, value in records.items():
                LOGGER.debug(key)
                for record in value:
                    # LOGGER.debug(record.value)
                    data.append(pickle.loads(record.value))

                return data
        except Exception as e:
            LOGGER.error(f"Something went wrong.. {e}")
            return []

    def pause(self):
        pass

    def resume(self):
        pass

    def close(self):
        LOGGER.debug("Shutting down consumer")
        self.consumer.close()
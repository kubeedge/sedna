import pickle
from json import dumps
from sedna.common.log import LOGGER
from sedna.datasources.kafka import *

POLL_TIMEOUT_MS = 5000

class Consumer(Client):
    def __init__(self, address = ["localhost"], port = [9092], group_id="default") -> None:
        super.__init__(address, port)
        self.consumer = KafkaConsumer(
            group_id=group_id,
            bootstrap_servers=self.kafka_endpoints,
            auto_offset_reset='earliest',
            enable_auto_commit=True
            )

    def get_topics(self):
        return self.consumer.topics()

    def subscribe(self, topic):
        try:
            self.consumer.subscribe(topic)
        except Exception as e:
             LOGGER.error(f"Unable to subscribe to topic {e}")

    # This should work with callbacks?
    def consume_json_data(self):
        data = []

        if len(self.consumer.subscription()) == 0:
            LOGGER.error("Not subscribed to any topic")
            return data

        try:
            LOGGER.info("Polling topics ...")
            records = self.consumer.poll(timeout_ms=POLL_TIMEOUT_MS)

            if len(records)==0:
                LOGGER.info("No message(s) consumed (maybe we timed out waiting?)\n")

            LOGGER.info("Consuming messages")
            for message in records:
                message = message.value

                data.append(pickle.loads(message))
                return data
        except Exception as e:
            LOGGER.error(f"Something went wrong.. {e}")
            return []

    def pause(self):
        pass

    def resume(self):
        pass

    def close(self):
        LOGGER.info("Shutting down consumer")
        self.consumer.close()
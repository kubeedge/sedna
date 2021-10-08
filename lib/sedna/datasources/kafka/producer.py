import pickle
from sedna.datasources.kafka import *

class Producer(Client):
    def __init__(self, address = ["localhost"], port = [9092], _metric_reporters=[]) -> None:
        super().__init__(address, port)
        LOGGER.debug("Creating Kafka producer")
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_endpoints,
            max_request_size=10000000,
            metric_reporters=_metric_reporters
            )

    def publish_data(self, data, topic = "default") -> bool:

        data_input = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        # Publishes a message
        for t in topic:
            try:
                self.producer.send(t, value=data_input)
                #self.producer.flush()
                LOGGER.debug(f"Message published on topic {topic}.")
                return True
            except Exception as e:
                LOGGER.error(f"Something went wrong.. {e}")
                return False


    def close(self):
        LOGGER.debug("Shutting down producer")
        self.producer.close()
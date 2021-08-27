import pickle

from sedna.datasources.kafka import *

class Producer(Client):
    def __init__(self, address = ["localhost"], port = [9092]) -> None:
        super().__init__(address, port)
        LOGGER.info("Creating Kafka producer")
        self.producer = KafkaProducer(bootstrap_servers=self.kafka_endpoints)

    def publish_data(self, data, topic = "default") -> bool:

        data_input = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        # Publishes a message
        try:
            self.producer.send(topic, value=data_input)
            self.producer.flush()
            LOGGER.info(f"Message published on topic {topic}.")
            return True
        except Exception as e:
            LOGGER.error(f"Something went wrong.. {e}")
            return False


    def close(self):
        LOGGER.info("Shutting down producer")
        self.producer.close()
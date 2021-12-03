from sedna.datasources.kafka.producer import Producer
from sedna.datasources.kafka.consumer import Consumer

from sedna.common.log import LOGGER

from threading import Thread

class KafkaProducer:
    def __init__(self, address, port, topic="default"):
        self.producer = Producer(address=address, port=port)
        self.topic = topic

        # This is not needed as usually brokers have auto.create.topics.enable = true
        # from sedna.datasources.kafka import AdminClient
        # ac = AdminClient(address, port)
        # try:
        #     ac.create_topics(self.topic)
        # except Exception as ex: # Should be TopicAlreadyExistsError?
        #     LOGGER.error(f"Topic already created - skipping error. [{ex}]")

    def write_result(self, data):
        return self.producer.publish_data(data, topic=self.topic)

class KafkaConsumerThread(Thread):
    def __init__(self, address, port, topic="default", sync_queue=None):
        super().__init__()
        self.consumer = Consumer(address=address, port=port)
        self.sync_queue = sync_queue
        self.topic = topic

        # We do this before actually reading from the topic
        self.consumer.subscribe(self.topic)

        self.daemon = True
        self.start()

    def run(self):
        while True:
            data = self.consumer.consume_json_data()
            if data:
                self.sync_queue.put(data)
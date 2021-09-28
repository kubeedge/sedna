import queue
from sedna.core.base import JobBase
from sedna.datasources.kafka.producer import Producer
from sedna.datasources.kafka.consumer import Consumer

from sedna.common.log import LOGGER

from threading import Thread

class KafkaProducer:
    # Address and port should come from the YAML
    # Or retrieved as cluster resources by the golang controller
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
    # Address and port should come from the YAML
    # Or retrieved as cluster resources by the golang controller
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

class KafkaBaseService(JobBase):
    """
    Base service using Kafka to transfer data between different pods/applications
    """

    def __init__(self, estimator=None, config=None, consumer_topics = [], producer_topics = []):
        super(KafkaBaseService, self).__init__(
            estimator=estimator, config=config)
        
        self.kafka_address = self.get_parameters("KAFKA_BIND_IPS", ["7.182.9.110"])
        self.kafka_port = self.get_parameters("KAFKA_BIND_PORTS", [32523])

        self.sync_queue = queue.Queue()

        if len(consumer_topics):
            self.consumer = KafkaConsumerThread(self.kafka_address, self.kafka_port, topic=consumer_topics, sync_queue=self.sync_queue)
        
        if len(producer_topics):
            self.producer = KafkaProducer(self.kafka_address, self.kafka_port, topic=producer_topics)


    def start(self):
        if callable(self.estimator):
            self.estimator = self.estimator()

        LOGGER.info("Creating sync_inference thread")
        self.sync_inference()

    def sync_inference(self):
        while True:
            token = self.sync_queue.get()
            LOGGER.info(f'Data consumed')
            try:
                self.inference(token)
            except Exception as e:
                LOGGER.info(f"Error processing token {token}: {e}")

            self.sync_queue.task_done()

    def inference(self, data=None, post_process=None, **kwargs):
        return
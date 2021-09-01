from sedna.common.log import LOGGER

from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka import KafkaAdminClient
from sedna.common.log import LOGGER

from kafka.admin import NewTopic

class Client():
    def __init__(self, address = ["localhost"], port = [9092]) -> None:
        self.kafka_address = address
        self.kafka_port =  port
        self.kafka_endpoints = []

        for addr, port in zip(self.kafka_address, self.kafka_port):
            LOGGER.debug(f"Adding address {addr} with port {port} to list of endpoints.")
            self.kafka_endpoints.append(f"{addr}:{port}")

class AdminClient(Client):
    def __init__(self, address = ["localhost"], port = [9092]) -> None:
        super().__init__(address, port)
        LOGGER.debug("Creating Kafka admin client")
        self.admin_client = KafkaAdminClient(bootstrap_servers=self.kafka_endpoints, client_id='test')


    def _parse_topics(self, topics, num_partitions, replication_factor):
        topic_list = []
        for topic in topics:
            topic_list.append(NewTopic(name=topic, num_partitions=num_partitions, replication_factor=replication_factor))

        return topic_list

    def create_topics(self, topics, num_partitions=1, replication_factor=1):
        topic_list = self._parse_topics(topics, num_partitions, replication_factor)
        res = self.admin_client.create_topics(new_topics=topic_list, validate_only=False)
        
        return res

    def delete_topics(self, topics, num_partitions=1, replication_factor=1):
        topic_list = self._parse_topics(topics, num_partitions, replication_factor)
        res = self.admin_client.delete_topics(new_topics=topic_list, validate_only=False)
        
        return res

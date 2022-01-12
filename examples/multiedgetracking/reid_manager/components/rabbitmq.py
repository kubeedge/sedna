import pika
import uuid
import json

from sedna.common.log import LOGGER

# TODO: Is it worth using a decorator with pre/post execution hooks?
class RabbitMQWriter():
    """
    Basic implementation of a RabbitMQ writer.
    """
    def __init__(self, address, port, queue='reid') -> None:
        self.uuid = uuid.uuid4().hex

        LOGGER.info(f"Start RAbbitMQQriter with UUID {self.uuid}")

        self.address = address
        self.port = port
        self.queue = queue

        LOGGER.info("Connecting to RabbitMQ server")
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, self.port))

    def target_lost(self):
        try:
            channel = self.connection.channel()
        except pika.exceptions.ConnectionWrongStateError:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, self.port))
            channel = self.connection.channel()
            
        channel.queue_declare(queue=self.queue)

        data = {
            "code":40001,		
            "content": {
                "msg":"TARGET_LOST"
            }
        }

        try:
            channel.basic_publish(exchange='', routing_key=self.queue, body=json.dumps(data))
            channel.close()
        except Exception as ex:
            LOGGER.error(f"Unable to write to RabbitMQ. [{ex}]")

    def target_found(self, address, userid, elem, index):
        LOGGER.debug("Writing to RabbitMQ")
        try:
            channel = self.connection.channel()
        except pika.exceptions.ConnectionWrongStateError:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, self.port))
            channel = self.connection.channel()

        channel.queue_declare(queue=self.queue)


        data = {
            "code":40002, 
            "content": {
            "historyLocation": [],
            "currentLocation": 
            {						
                "pullStream": address,	
                "cameraId": elem.camera[0],					
                "coordinate":"0.001497,0.000002",
                "seq": index		
            },
            "userId": userid,
            "trackId": self.uuid,			
            "msg":"SEARCH_SUCCESS"			 	 
            }
        }

        try:
            channel.basic_publish(exchange='', routing_key=self.queue, body=json.dumps(data))
            channel.close()
        except Exception as ex:
            LOGGER.error(f"Unable to write to RabbitMQ. [{ex}]")

    def search_failure(self):
        pass 
from abc import ABC, abstractmethod
import distutils
from sedna.common.log import LOGGER

#TODO:
# how to get topic names?
# how to get the right server and client instance?

class BaseService(ABC):
    """
    Base service
    """

    def __init__(self):
        self.log = LOGGER
        self.kafka_enabled = bool(distutils.util.strtobool(self.get_parameters("KAFKA_ENABLED", "False")))

        self.api_ip = self.get_parameters("MANAGER_API_BIND_IP", "7.182.9.110")
        self.api_port = int(self.get_parameters("MANAGER_API_BIND_PORT", "27345"))

        # If we explicity define the endpoints in the YAML file, we do not use the API manager pod
        self.upload_endpoint = self.get_parameters("upload_endpoint", f"http://{self.api_ip}:{self.api_port}/sedna/upload_data")
        self.status_update_endpoint = self.get_parameters("status_update_endpoint", f"http://{self.api_ip}:{self.api_port}/sedna/get_app_details")
        self.post_process_result = bool(distutils.util.strtobool(self.get_parameters("post_process_result", "False")))

        if self.kafka_enabled:
            self.log.debug("Kafka support enabled in YAML file")
            self.kafka_address = self.get_parameters("KAFKA_BIND_IPS", ["7.182.9.110"])
            self.kafka_port = self.get_parameters("KAFKA_BIND_PORTS", [32669])

            if isinstance(self.kafka_address, str):
                self.log.debug(f"Parsing string received from K8s controller {self.kafka_address},{self.kafka_port}")
                self.kafka_address = self.kafka_address.split("|")
                self.kafka_port = self.kafka_port.split("|")
            
            self.sync_queue = queue.Queue()

            self.producer = KafkaProducer(self.kafka_address, self.kafka_port, topic=["reid"])
            self.consumer = KafkaConsumerThread(self.kafka_address, self.kafka_port, topic=["feature_extraction"], sync_queue=self.sync_queue)

        self.op_mode = "detection"

    @abstractmethod
    def start(self):
        if callable(self.estimator):
            self.estimator = self.estimator()

        StatusSyncThread(self.update_operational_mode, self.status_update_endpoint)

        # The cloud instance only runs a distance function to do the ReID
        # We don't load any model here.
        if self.kafka_enabled:
            self.log.debug("Creating sync_inference thread")
            self.fetch_data()
        else:
            self.log.debug("Starting default REST webservice")
            app_server = ReIDServer(model=self, servername=self.job_name, host=self.local_ip, http_port=self.port)
            app_server.start()

    @abstractmethod
    def fetch_data(self):
        while True:
            token = self.sync_queue.get()
            self.log.debug(f'Data consumed')
            try:
                self.inference(token)
            except Exception as e:
                self.log.debug(f"Error processing received data: {e}")

            self.sync_queue.task_done()
    
    @abstractmethod
    def inference(self, data=None, post_process=None, **kwargs):
        pass
    
    @abstractmethod
    def update_operational_mode(self, status):
        pass
import time, json

from sedna.common.config import Context
from sedna.common.log import LOGGER

# Could be added in a ConfigMap
FLUENTD_ADDRESS = Context.get_parameters("FLUENTD_IP", None)
FLUENTD_PORT = 24224
SEDNA_INDEX = 'sedna'

from fluent import sender
from fluent import event

# Base class to send events to the Fluentd daemon in the cluster (if available)
class FluentdHelper():
    def __init__(self, index=SEDNA_INDEX):
        super().__init__()
        if FLUENTD_ADDRESS:
            # 'sedna' is a dedicated index in ES
            sender.setup(SEDNA_INDEX, host=FLUENTD_ADDRESS, port=FLUENTD_PORT)
    
    # msg must be a json dict (e.g, {'valA' : 1 ..})
    def send_json_msg(self, msg):
        if FLUENTD_ADDRESS:
            event.Event('follow', {'message': json.dumps(msg)})
        

# Context Manager class to measure exeuction time of a function
class FTimer(FluentdHelper):
    def __init__(self, name="", extra=None):
        super(FTimer, self).__init__()
        self.start = time.time()
        self.log = LOGGER
        self.name = name

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start

        result = {
            "execution_time": runtime,
            "method_name": self.name
        }

        if FLUENTD_ADDRESS:
            self.send_json_msg(result)
        self.log.debug(json.dumps(result))
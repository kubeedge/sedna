import time, json
from sedna.common.config import Context
from sedna.common.log import LOGGER

# This belong to a ConfigMap
FLUENTD_ADDRESS = Context.get_parameters("FLUENTD_IP", None)
FLUENTD_PORT = 24224
FLUENTD_INDEX = 'sedna'

if FLUENTD_ADDRESS:
    from fluent import sender
    from fluent import event

class FTimer():
    def __init__(self, name=""):
        self.start = time.time()
        self.log = LOGGER
        self.name = name

        if FLUENTD_ADDRESS:
            # 'sedna' is a dedicated index in ES
            sender.setup(FLUENTD_INDEX, host=FLUENTD_ADDRESS, port=FLUENTD_PORT)

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
            # The parameters "follow" and "message" are connected to the 
            # filter/parser in the fluentd configuration we created for Sedna
            event.Event('follow', {'message': json.dumps(result)})

        self.log.debug(json.dumps(result))

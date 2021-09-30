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

class FluentdHelper():
    def __init__(self):
        if FLUENTD_ADDRESS:
            # 'sedna' is a dedicated index in ES
            sender.setup(FLUENTD_INDEX, host=FLUENTD_ADDRESS, port=FLUENTD_PORT)
    
    # msg must be a json dict (e.g, {'valA' : 1 ..})
    def send_json_msg(self, msg):
        event.Event('follow', {'message': json.dumps(msg)})

class FTimer(FluentdHelper):
    def __init__(self, name="", extra=None):
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

        self.send_json_msg(result)
        self.log.debug(json.dumps(result))

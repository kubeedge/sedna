# from threading import Thread
import time, json
# import psutil
from sedna.common.config import Context
from sedna.common.log import LOGGER

# This belong to a ConfigMap
FLUENTD_ADDRESS = Context.get_parameters("FLUENTD_IP", None)
FLUENTD_PORT = 24224
SEDNA_INDEX = 'sedna'

if FLUENTD_ADDRESS:
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

        self.send_json_msg(result)
        self.log.debug(json.dumps(result))

# Class to monitor resource utlization of a pod belonging to Sedna
# class ResourceMonitor(FluentdHelper, Thread):
#     def __init__(self, interval = 1, refresh_interval = 0.5) -> None:
#         super().__init__()
        
#         self.interval = interval
#         self.refresh_interval = refresh_interval
#         self.daemon = True
        
#         self.start()

#     def run(self):
#         LOGGER.debug("Start ResourceMonitor thread")
#         while True:
#             self.collect()
#             time.sleep(self.refresh_interval)

#     def collect(self):
#         data = {
#             "cpu%": psutil.cpu_percent(),
#             "mem%": psutil.virtual_memory().percent,
#             "mem_available": psutil.virtual_memory().available,
#             "mem_used": psutil.virtual_memory().used,
#             "mem_total": psutil.virtual_memory().used,
#             "net_bytes_sent": psutil.net_io_counters().bytes_sent,
#             "net_bytes_recv": psutil.net_io_counters().bytes_recv,
#         }
        
#         self.send_json_msg(data)

import time, json
from sedna.common.config import BaseConfig
from sedna.common.log import LOGGER

FLUENTD_ADDRESS = BaseConfig.fluentd_address

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
            sender.setup('sedna', host=FLUENTD_ADDRESS, port=24224)

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

# if __name__ == '__main__':
#     with MyTimer():
#         long_runner()


# def benchmark_time(func):
#     """
#     A timer decorator
#     """
#     def function_timer(*args, **kwargs):
#         """
#         A nested function for timing other functions
#         """
#         start = time.time()
#         value = func(*args, **kwargs)
#         end = time.time()
#         runtime = end - start
#         msg = "{func}:{time}s"
#         LOGGER.info(msg.format(func=func.__name__,
#                          time=runtime))
#         return value
#     return function_timer


import random
import time

from sedna.common.log import LOGGER

class FTimer():
    def __init__(self, name=""):
        self.start = time.time()
        self.log = LOGGER
        self.name = name
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        self.log.info(f"{self.name}:{runtime}s")

# if __name__ == '__main__':
#     with MyTimer():
#         long_runner()


def benchmark_time(func):
    """
    A timer decorator
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "{func}:{time}s"
        self.log.info(msg.format(func=func.__name__,
                         time=runtime))
        return value
    return function_timer


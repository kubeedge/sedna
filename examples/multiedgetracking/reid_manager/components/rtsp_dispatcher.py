import threading
from sedna.common.log import LOGGER

lock = threading.Lock()

streams = [
    "rtsp://172.17.0.1/video/0",
    "rtsp://172.17.0.1/video/1",
    "rtsp://172.17.0.1/video/2"
    ]

def get_rtsp_stream():
    LOGGER.debug("Waiting for a lock")
    lock.acquire()
    item = None

    if len(streams) > 0: 
        try:
            LOGGER.debug('Acquired a lock, popping stream URI')
            item = streams.pop(0)
            # This is a dirty way to make the list circular ..
            streams.append(item) 

        finally:
            LOGGER.debug('Released a lock')
            lock.release()
    
    return str(item)

# Currently not used
def add_rtsp_stream(item : str):
    LOGGER.debug("Waiting for a lock")
    lock.acquire()

    if item: 
        try:
            LOGGER.debug('Acquired a lock, adding new stream URI')
            streams.append(item) 
        finally:
            LOGGER.debug('Released a lock')
            lock.release()
    
        return 200
    else:
        return 404
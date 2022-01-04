import threading, json
from sedna.common.log import LOGGER

lock = threading.Lock()

streams = []

def get_rtsp_stream():
    res = {}
    
    try:
        LOGGER.debug("Waiting for a lock")
        lock.acquire()
        if len(streams) > 0:
            LOGGER.debug('Acquired a lock, popping stream URI')
            item = streams.pop(0)

            res = {
                "camera_address": item[0],
                "camera_id": item[1]
            }

        else:
            LOGGER.warning("No URIs available.")
    except Exception as ex:
        LOGGER.error(f"Error while fetching the RTSP stream. [{ex}]")
    finally:
        LOGGER.debug('Releasing lock')
        lock.release()
    
    return json.dumps(res)

def add_rtsp_stream(item : str, camid: int):
    LOGGER.debug("Waiting for a lock")
    if item: 
        try:
            LOGGER.debug('Acquired a lock, adding new stream URI')
            lock.acquire()
            streams.append([item, camid]) 
        finally:
            LOGGER.debug('Released a lock')
            lock.release()
    
        return 200
    else:
        return 404

def reset_rtsp_stream_list():
    try:
        LOGGER.debug('Clearing the list of streams')
        lock.acquire()
        streams.clear()
    finally:
        lock.release()

    return 200

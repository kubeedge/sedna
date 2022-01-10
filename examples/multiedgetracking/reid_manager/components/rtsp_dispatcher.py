import threading, json
from typing import List
from sedna.common.log import LOGGER

lock = threading.Lock()

class RTSPEntry():
    def __init__(self, address, camera_id, receiver="unknown") -> None:
        self.address = address
        self.camera_id = camera_id
        self.receiver = receiver

streams : List[RTSPEntry] = []

def get_rtsp_stream(hostname):
    res = {}
    
    try:
        LOGGER.debug("Waiting for a lock")
        lock.acquire()
        if len(streams) > 0:
            LOGGER.debug('Acquired a lock, popping stream URI')         
            item_list = list(filter(lambda x: hostname.strip().replace(" ","") == x.receiver, streams))

            # In the future, this list might contain multiple elements.
            if len(item_list) > 0:

                res = {
                    "camera_address": item_list[0].address,
                    "camera_id": item_list[0].camera_id
                }

                streams.remove(item_list[0])

        else:
            LOGGER.warning("No URIs available.")
    except Exception as ex:
        LOGGER.error(f"Error while fetching the RTSP stream for hostname {hostname}. [{ex}]")
    finally:
        LOGGER.debug('Releasing lock')
        lock.release()
    
    return json.dumps(res)

def add_rtsp_stream(item : str, camid: int, receiver="unknown"):
    LOGGER.debug("Waiting for a lock")
    if item: 
        try:
            LOGGER.debug('Acquired a lock, adding new stream URI')
            lock.acquire()
            streams.append(RTSPEntry(item, camid, receiver)) 
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

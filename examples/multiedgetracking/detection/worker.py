import json
import time
import numpy

import requests
import cv2
import threading

from sedna.algorithms.optical_flow import LukasKanade
from sedna.core.multi_edge_tracking import ObjectDetector

from sedna.common.config import Context
from sedna.common.log import LOGGER

from estimator import str_to_estimator_class

camera_address = Context.get_parameters('video_url')
stream_dispatcher = Context.get_parameters('stream_dispatcher_url')
estimator_class = Context.get_parameters('estimator_class', "Yolov5")

def retrieve_rtsp_stream() -> str:
    LOGGER.debug(f'Finding target RTSP stream')
    if stream_dispatcher != None:
        try:
            rtsp_stream = requests.get(stream_dispatcher)
            LOGGER.debug(f'Retrieved RTSP stream with address {rtsp_stream}')
            # We have to do this sanity check otherwise cv2 will silenty fail and never open the RTSP stream
            cv2_cleaned_string = rtsp_stream.text.strip().replace('"', '')
            return cv2_cleaned_string
        except Exception as ex:
            LOGGER.error(f'Unable to access stream dispatcher server, using fallback value. [{ex}]')
            return camera_address
    else:
        LOGGER.debug(f'Using RTSP from env variable with address {camera_address}')
        return camera_address
    

def start_stream_acquisition(stream_address):
    optical_flow = LukasKanade()
    camera_code = stream_address.split("/")[-1] # WARNING: Only for demo purposes!
    eclass = str_to_estimator_class(estimator_class=estimator_class)
    selected_estimator=eclass(camera_code=camera_code)
    edge_worker = ObjectDetector(selected_estimator)

    camera = cv2.VideoCapture(stream_address)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    fps = 0.5
    nframe = 0
    startTime = time.time()
    prev_frame = numpy.empty(0)

    while True:
        try:
            # We use grab() to avoid populating the camera buffer with images from the past.
            # If the detection would be fast enough, we would not need this workaround.
            # Also, this should be event based rather than using time intervals.
            ret = camera.grab()
            nowTime = time.time()
        except Exception as ex:
            LOGGER.error(f'Unable to access stream [{ex}]')
    
        if not ret:
            LOGGER.debug(
                f" Camera is not open, camera_address={stream_address},"
                f" Sleep 5 second.")
            time.sleep(5)
            try:
                camera = cv2.VideoCapture(stream_address)
            except Exception as ex:
                LOGGER.error(f'Unable to access stream [{ex}]')
            continue

        if nframe % fps:
            nframe += 1
            continue


        if nowTime - startTime > 1/fps:
            ret, input_yuv = camera.read()
            img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_BGR2RGB)
            LOGGER.debug(f"Camera is open, current frame index is {nframe}")

            if prev_frame.size:
                if optical_flow(prev_frame, img_rgb):
                    LOGGER.info("Movement detected")
                    threading.Thread(target=edge_worker.inference, args=(img_rgb,), daemon=False).start()
            else:
                # The first time we are going to process the frame anyway
                LOGGER.info("Processing first frame in the RTSP stream")
                threading.Thread(target=edge_worker.inference, args=(img_rgb,), daemon=False).start()
            
            nframe += 1
            prev_frame = img_rgb
            startTime = time.time() # reset time

if __name__ == '__main__':
    result = retrieve_rtsp_stream()
    start_stream_acquisition(result)
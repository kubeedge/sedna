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

class Bootstrapper():
    def __init__(self) -> None:
        LOGGER.info("Creating Detection/Tracking Bootstrapper module")

        self.api_ip = Context.get_parameters("MANAGER_API_BIND_IP", "7.182.9.110")
        self.api_port = int(Context.get_parameters("MANAGER_API_BIND_PORT", "27345"))
        self.stream_dispatcher = Context.get_parameters('stream_dispatcher_url', f"http://{self.api_ip}:{self.api_port}/sedna/get_video_address")
        self.estimator_class = Context.get_parameters('estimator_class', "Yolov5")
        self.camera_address = Context.get_parameters('video_url')
        self.optical_flow = LukasKanade()

        self.eclass = str_to_estimator_class(estimator_class=self.estimator_class)
        
        self.camera_id = -1
        self.camera_address = None

    # Right now, we assume that we process only one video stream
    def run(self):
        LOGGER.info("Acquiring camera feed")
        # TODO: Access using a key (e.g., UUID of the detector) to retrieve a LIST of
        # RTSP streams. Then, start a thread for each stream.
        while self.camera_id == -1 and self.camera_address == None:
            self.camera_address, self.camera_id = self.retrieve_rtsp_stream()
            time.sleep(2)

        self.start_stream_acquisition(self.camera_address, self.camera_id)

    def retrieve_rtsp_stream(self) -> str:
        LOGGER.debug(f'Retrieving source stream/s')
        try:
            rtsp_stream = requests.get(self.stream_dispatcher)
            data = json.loads(rtsp_stream.json())
            # We have to do this sanity check otherwise cv2 will silenty fail and never open the RTSP stream
            address = data["camera_address"].strip().replace('"', '')
            camera_id = data["camera_id"]

            LOGGER.info(f'Retrieved video source details: {data}')

            return address, camera_id
        except Exception as ex:
            LOGGER.error(f'Unable to retrieve source stream/s. [{ex}]')
            return None, -1

    def start_stream_acquisition(self, stream_address, camera_code):
        selected_estimator=self.eclass(camera_code=camera_code)
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
                    if self.optical_flow(prev_frame, img_rgb):
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
    bs = Bootstrapper()
    bs.run()
import json
import time
import numpy

import requests
import cv2
import torch

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
        self.hostname = Context.get_parameters('hostname', "unknown")
        self.fps = float(Context.get_parameters('fps', 1))
        self.optical_flow = LukasKanade()

        self.eclass = str_to_estimator_class(estimator_class=self.estimator_class)
        
        self.camera_id = -1
        self.camera_address = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parallelism = 6

    # Right now, we assume that we process only one video stream
    def run(self):
        LOGGER.info("Acquiring camera feed")
        # TODO: Access using a key (e.g., UUID of the detector) to retrieve a LIST of
        # RTSP streams. Then, start a thread for each stream.
        while self.camera_id == -1 and self.camera_address == None:
            self.camera_address, self.camera_id = self.retrieve_rtsp_stream()
            time.sleep(2)

        getattr(self, f"start_stream_acquisition_{self.device}")(self.camera_address, self.camera_id)

    def retrieve_rtsp_stream(self) -> str:
        LOGGER.debug(f'Retrieving source stream/s')
        try:
            rtsp_stream = requests.get(self.stream_dispatcher, params={"receiver":self.hostname}, timeout=5)
            data = json.loads(rtsp_stream.json())
            # We have to do this sanity check otherwise cv2 will silenty fail and never open the RTSP stream
            address = data["camera_address"].strip().replace('"', '')
            camera_id = data["camera_id"]

            LOGGER.info(f'Retrieved video source details: {data}')

            return address, camera_id
        except Exception as ex:
            LOGGER.error(f'Unable to retrieve source stream/s. [{ex}]')
            return None, -1

    def connect_to_camera(self, stream_address):
        camera = None
        while camera == None or not camera.isOpened():
            try:
                camera = cv2.VideoCapture(stream_address, cv2.CAP_FFMPEG)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            except Exception as ex:
                LOGGER.error(f'Unable to access stream [{ex}]')
            time.sleep(1)

        return camera

    def start_stream_acquisition_cuda(self, stream_address, camera_code):
        selected_estimator=self.eclass(camera_code=camera_code)
        nframe = 0

        startTime = time.time()
        prev_frame = numpy.empty(0)

        # We use a pool of workers with GPU
        worker_pool = [ObjectDetector(selected_estimator)] * self.parallelism

        # First connection to the camera/video stream
        camera = self.connect_to_camera(stream_address)

        while True:
            # It can happen that the camera closes at somepoint or the video interrputs.
            # In such cases, we need to reconnect.
            if not camera.grab():
                camera = self.connect_to_camera(stream_address)

            nowTime = time.time()
            
            if nowTime - startTime > 1/self.fps:
                try:
                    grabbed, input_yuv = camera.read()
                    if grabbed:
                        nframe += 1
                        img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_BGR2RGB)
                        LOGGER.debug(f"Camera is open, current frame index is {nframe}")

                        if prev_frame.size:
                            if self.optical_flow(prev_frame, img_rgb):
                                LOGGER.debug("Movement detected")
                                worker_pool[nframe % len(worker_pool)].put_data(img_rgb)
                                # threading.Thread(target=edge_worker.inference, args=(img_rgb,), daemon=False).start()
                        else:
                            # The first time we are going to process the frame anyway
                            LOGGER.debug("Processing first frame in the RTSP stream")
                            worker_pool[nframe % len(worker_pool)].put_data(img_rgb)
                            # threading.Thread(target=edge_worker.inference, args=(img_rgb,), daemon=False).start()
                        
                        # nframe += 1
                        prev_frame = img_rgb
                except Exception as ex:
                    LOGGER.error(ex)
                
                startTime = time.time() # reset time

    def start_stream_acquisition_cpu(self, stream_address, camera_code):
        selected_estimator=self.eclass(camera_code=camera_code)
        edge_worker = ObjectDetector(selected_estimator)

        fps = self.fps
        nframe = 0
        startTime = time.time()
        prev_frame = numpy.empty(0)

        camera = self.connect_to_camera(stream_address)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        while True:           
            # It can happen that the camera closes at somepoint or the video interrputs.
            # In such cases, we need to reconnect.
            if not camera.grab():
                camera = self.connect_to_camera(stream_address)
            
            # We use grab() to avoid populating the camera buffer with images from the past.
            # If the detection would be fast enough, we would not need this workaround.
            # Also, this should be event based rather than using time intervals.
            nowTime = time.time()
        
            if nowTime - startTime > 1/fps:
                _, input_yuv = camera.read()
                img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_BGR2RGB)
                LOGGER.debug(f"Camera is open, current frame index is {nframe}")

                if prev_frame.size:
                    if self.optical_flow(prev_frame, img_rgb):
                        LOGGER.info("Movement detected")
                        edge_worker.put_data(img_rgb)
                        # threading.Thread(target=edge_worker.inference, args=(img_rgb,), daemon=False).start()
                else:
                    # The first time we are going to process the frame anyway
                    LOGGER.info("Processing first frame in the RTSP stream")
                    edge_worker.put_data(img_rgb)
                    # threading.Thread(target=edge_worker.inference, args=(img_rgb,), daemon=False).start()
                
                nframe += 1
                prev_frame = img_rgb
                startTime = time.time() # reset time

if __name__ == '__main__':
    bs = Bootstrapper()
    bs.run()
import threading
import time
import cv2
from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult

class VideoBuilder(threading.Thread):
    def __init__(self, reid_buffer, rtmp_url, user_id, fps=2, height=480, width=640) -> None:
        super().__init__()
        
        LOGGER.info(f"Initializing VideoBuilder thread to stream to {rtmp_url}{user_id}")

        self.reid_buffer = reid_buffer

        self.rtmp_url = rtmp_url
        self.user_id = user_id

        # Output stream specs (default values are optimal)
        self.fps = fps
        self.height = height
        self.width = width

        self.daemon = True

        self.start()

    def _create_pipe(self):
        import subprocess as sp
        LOGGER.info("Create FFMPEG pipe")
        container = "rtsp"

        if self.rtmp_url.split("::")[0] == "rtmp":
            LOGGER.info("Using FLV protocol")
            container = "flv"
        else:
            LOGGER.info("Using RTSP protocol")
            container = "rtsp"

        command = ['ffmpeg',
                '-loglevel', 'error',
                '-c',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f"{self.width}x{self.height}",		# width and height of your image
                '-r', str(self.fps),
                '-i', '-',	
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'veryfast',
                '-stimeout', '50000',
                '-f', f'{container}',
                self.rtmp_url+self.user_id]
               
        pipe_push = sp.Popen(command, stdin=sp.PIPE, shell=False)

        return pipe_push
    
    def _recover_pipe(self, pipe):
        LOGGER.debug("Terminate broken pipe")
        pipe.stdin.close()
        pipe.kill()

        return self._create_pipe()

    def run(self) -> None:
        pipe = self._create_pipe()
        image = None

        LOGGER.info("Start streaming loop")
        while True:
            try:
                if len(self.reid_buffer.frame_buffer) > 0:
                    frame : DetTrackResult = self.reid_buffer.frame_buffer.popleft()
                    image = cv2.imdecode(frame.scene, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (640,480))
                    
                    LOGGER.debug("Writing into output stream")
                    pipe.stdin.write(image)
                else:
                    if image is not None:
                        pipe.stdin.write(image)
                    time.sleep(1)

            except Exception as ex:
                LOGGER.error(f"Error during transmission to RTMP server. [{ex}]")
                pipe = self._recover_pipe(pipe)
import threading
import time
import cv2
from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult

class VideoBuilder(threading.Thread):
    def __init__(self, reid_buffer, rtmp_url, user_id, fps) -> None:
        super().__init__()
        
        self.reid_buffer = reid_buffer

        self.rtmp_url = rtmp_url
        self.user_id = user_id
        self.fps = fps
        
        LOGGER.info(f"Initializing VideoBuilder thread to stream to {self.rtmp_url}{self.user_id}")

        self.daemon = True

        self.start()

 # Called only once to create the pipe to write to FFMPEG
    # TODO: Use python-ffmpeg library rather than subprocess..
    def _create_rtmp_pipe(self, height=480, width=640):
        import subprocess as sp
        LOGGER.debug("Create RTMP pipe")
        
        if self.rtmp_url.split("::")[0] == "rtmp":
            container = "flv"
        else:
            container = "rtsp"

        command = ['ffmpeg',
                '-loglevel', 'error',
                '-c',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f"{width}x{height}",		# weight and height of your image
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
    
    def run(self) -> None:
        LOGGER.info("Start streaming")
        pipe = self._create_rtmp_pipe()
        image = None

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
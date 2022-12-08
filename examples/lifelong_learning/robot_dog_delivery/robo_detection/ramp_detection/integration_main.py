# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import threading
import numpy as np
import multiprocessing
from cv_bridge import CvBridge
from robosdk.core import Robot
from robosdk.utils.context import Context
from robosdk.utils.constant import GaitType
from robosdk.msgs.sender.ros import RosMessagePublish

from ramp_detection.integration_interface import Estimator

class Detection:
    def __init__(self):
        self.robot = Robot(name="x20", config="ysc_x20")
        self.segment = Estimator()
        self.robot.connect()
        self.publish = RosMessagePublish()
        _topic = Context.get("curb_detection", "/robovideo")
        self.publish.register("curb_detection", topic=_topic,
                              converter=CvBridge().cv2_to_imgmsg,
                              convert_param={"encoding": "bgr8"})
        # self.robot.navigation.speed(0.4)
        self.stair_start_time = None
        self.stair_hold_time = 7
        self.complete_change_upstair_time = None
        self.unseen_sample_threshold = 3
        self.unseen_sample_num = 0

    def run(self):
        if not getattr(self.robot, "camera", ""):
            return
        while 1:
            img, dep = self.robot.camera.get_rgb_depth()
            if img is None:
                continue

            result = self.segment.predict(img, depth=dep)
            if not result:
                self.robot.logger.info("Unseen sample detected.")
                # self.unseen_sample_num += 1
                # if self.unseen_sample_num >= self.unseen_sample_threshold:
                #    self.robot.logger.info("Stop in front of unseen sample!")  
                #    for _ in range(3):
                #        self.robot.navigation.stop()
                #    self.unseen_sample_num = 0
                continue
            # else:
            #    self.unseen_sample_num = 0
                  
            if result == "no_ramp":
                self.process_curb(result, img)
            else:
                self.process_ramp(result)

    def process_curb(self, result, img):
        current_time = time.time()
        if self.stair_start_time and current_time - self.stair_start_time < self.stair_hold_time:
            return
        elif self.stair_start_time and current_time - self.stair_start_time >= self.stair_hold_time:
            self.stair_start_time = None

        if self.segment.curr_gait == "up-stair":
            self.robot.logger.info("match curb")
            gait = threading.Thread(name="", target=self.change_to_upstair)
            stop = threading.Thread(name="", target=self.stop)
            gait.start()
            stop.start()
            self.stair_start_time = time.time()
        elif self.segment.curr_gait == "trot":
            self.robot.logger.info("unmatch curb")
            self.robot.control.change_gait(GaitType.TROT)

    def process_ramp(self, location):
        if location == "small_trapezoid":
            self.robot.logger.info(f"Ramp location: {location}. Keep moving.")
            self.robot.navigation.go_forward()
            return

        self.robot.logger.info(f"Ramp detected: {location}.")

        if location == "upper_left" or location == "center_left":
            self.robot.logger.info("Move to the left!")
            self.robot.navigation.turn_left()

        elif location == "bottom_left":
            self.robot.logger.info("Backward and move to the left!")
            self.robot.navigation.go_backward()
            self.robot.navigation.turn_left()

        elif location == "upper_right" or location == "center_right":
            self.robot.logger.info("Move to the right!")
            self.robot.navigation.turn_right()

        elif location == "bottom_right":
            self.robot.logger.info("Backward and move to the right!")
            self.robot.navigation.go_backward()
            self.robot.navigation.turn_right()

        self.robot.navigation.go_forward()

    def change_to_upstair(self):
        self.robot.control.change_gait(GaitType.UPSTAIR)
        self.complete_change_upstair_time = time.time()
        # print("complete_time:", self.complete_change_upstair_time)

    def stop(self):
        print("Stop in front of curb!")
        while 1:
            self.robot.navigation.stop()
            if self.complete_change_upstair_time is not None:
                break
        self.complete_change_upstair_time = None   



if __name__ == '__main__':
    project = Detection()
    project.run()

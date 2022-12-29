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

from cv_bridge import CvBridge
from robosdk.core import Robot
from robosdk.utils.context import Context
from robosdk.msgs.sender.ros import RosMessagePublish

from ramp_detection.interface import Estimator


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

    def run(self):
        if not getattr(self.robot, "camera", ""):
            return
        while 1:
            img, dep = self.robot.camera.get_rgb_depth()
            if img is None:
                continue
            result = self.segment.predict(img, depth=dep)
            print(result)

if __name__ == '__main__':
    project = Detection()
    project.run()

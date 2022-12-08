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
"""
Copyright (c) Deep Robotics Inc. - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Author: Haoyi Han <hanhaoyi@deeprobotics.cn>, Feb, 2020
"""

import time
import socket
import struct
import threading
from typing import Union

from robosdk.utils.class_factory import ClassFactory
from robosdk.utils.class_factory import ClassType
from robosdk.utils.util import Config
from robosdk.utils.constant import GaitType

from .base import LeggedController

__all__ = ("DeepRoboticsControl",)


class RobotCommander:
    """
    RobotCommander is resposible for command the robot to stop,tread or play.
    """

    _command_code = {
        "STAND_UP_DOWN": 1,
        "START_FORCE_MODE": 2,
        "MOTION_START_STOP": 3,
        "DANCE": 19,
        "CHANGE_GAIT": 25,
        "HEART_BEAT": 33
    }

    def __init__(self,
                 local_port=20001,
                 ctrl_ip='192.168.1.120',
                 ctrl_port=43893):
        self.local_port = local_port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.ctrl_addr = (ctrl_ip, ctrl_port)
        self.comm_lock = threading.Lock()

    def __enter__(self):
        self.server.bind(('0.0.0.0', self.local_port))
        self._keep_alive = True

        self.keep_alive_thread = threading.Thread(target=self.keep_alive,
                                                  name="keep_alive")
        self.keep_alive_thread.setDaemon(True)
        self.keep_alive_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.server = None
        self._keep_alive = False
        self.keep_alive_thread.join()

    def keep_alive(self):
        while self._keep_alive:
            self.sendSimpleCommand("HEART_BEAT", verbose=False)
            time.sleep(0.25)

    def sendSimple(self, command_code=25, command_value=0, command_type=0):
        data = struct.pack('<3i', command_code, command_value, command_type)
        self.comm_lock.acquire()
        self.server.sendto(data, self.ctrl_addr)
        self.comm_lock.release()

    def sendSimpleCommand(self, command_name, verbose=True):
        self.sendSimple(self._command_code[command_name])

    def stand_down_up(self):
        self.sendSimpleCommand("STAND_UP_DOWN")

    def dance(self):
        self.sendSimpleCommand("DANCE")

    def start_force_mode(self):
        self.sendSimpleCommand("START_FORCE_MODE")

    def motion_start_stop(self):
        self.sendSimpleCommand("MOTION_START_STOP")

    def yaw_adjust(self, adjust_rad):
        self.sendSimple(33, int(adjust_rad * 1000))

    def up_stair_trait(self):
        self.sendSimple(7)

    def finish_up_stair_trait(self):
        self.sendSimple(7)

    def down_stair_trait(self):
        self.sendSimple(7)
        time.sleep(0.1)
        self.sendSimple(2)

    def finish_down_stair_trait(self):
        self.sendSimple(2)
        time.sleep(0.1)
        self.sendSimple(7)


@ClassFactory.register(ClassType.CONTROL, alias="ysc_control")
class DeepRoboticsControl(LeggedController):  # noqa

    def __init__(self, name: str = "ysc", config: Config = None):
        super(DeepRoboticsControl, self).__init__(name=name, config=config)

        import rostopic
        import message_filters

        self._GAIT_CODE = {
            11: GaitType.LIEON,
            3: GaitType.RUN,
            2: GaitType.FALL,
            0: GaitType.TROT,
            1: GaitType.UPSTAIR,
        }
        self.msg_lock = threading.RLock()
        self.curr_gait = GaitType.UNKONWN
        self.commander = RobotCommander(
            local_port=self.config.parameter.local_port,
            ctrl_port=self.config.parameter.ctrl_port,
            ctrl_ip=self.config.parameter.ctrl_ip,
        )
        msg_type, _, _ = rostopic.get_topic_class(
            "/robot_gait_state")
        self.gait_sub = message_filters.Subscriber(
            "/robot_gait_state", msg_type
        )
        self.gait_sub.registerCallback(self.gait_listen)

    def gait_listen(self, msg):
        self.msg_lock.acquire()
        if not msg:
            self.curr_gait = GaitType.UNKONWN
        else:
            data = int(msg.data)
            if data in self._GAIT_CODE:
                self.curr_gait = self._GAIT_CODE[data]
            else:
                try:
                    self.curr_gait = GaitType(data)
                except ValueError:
                    self.curr_gait = GaitType.UNKONWN
        self.msg_lock.release()

    def get_curr_gait(self) -> GaitType:
        return self.curr_gait

    def change_gait(self, gait: Union[str, GaitType]):
        if isinstance(gait, str):
            gait = getattr(GaitType, gait.upper())
        now_gait = self.get_curr_gait()
        if now_gait == gait:
            return
        try_times = len(GaitType)
        while try_times:
            self.logger.info(f"change gait to {gait.name}, now {now_gait.name}")

            self.commander.sendSimple()
            time.sleep(1.0)
            now_gait = self.get_curr_gait()
            if now_gait == gait:
                break
            try_times -= 1
            # time.sleep(0.2)

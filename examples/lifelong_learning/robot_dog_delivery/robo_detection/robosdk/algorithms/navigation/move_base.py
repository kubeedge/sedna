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
from importlib import import_module

import numpy as np

from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory
from robosdk.utils.util import Config
from robosdk.utils.schema import BasePose
from robosdk.utils.schema import PathNode
from robosdk.utils.constant import ActionStatus
from robosdk.cloud_robotics.task_agent.base import RoboActionHandle

from .base import Navigation

__all__ = ("MoveBase",)


@ClassFactory.register(ClassType.NAVIGATION)
class MoveBase(Navigation):  # noqa

    def __init__(self,
                 name: str = "MoveBase",
                 config: Config = None
                 ):
        super(MoveBase, self).__init__(name=name, config=config)

        import rospy
        import actionlib
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import LaserScan
        from move_base_msgs.msg import MoveBaseAction
        from actionlib_msgs.msg import GoalStatusArray, GoalID

        self.move_base_ac = actionlib.SimpleActionClient(
            self.config.target.action, MoveBaseAction
        )
        rospy.Subscriber(
            self.config.target.status,
            GoalStatusArray,
            self._move_base_status_callback,
        )
        self.scan_msg = LaserScan()
        rospy.Subscriber(
            self.config.target.laserscan, LaserScan,
            self._laser_scan_callback)
        self.move_base_cancel = rospy.Publisher(
            self.config.target.cancel, GoalID, queue_size=1
        )
        self.curr_vel = BasePose()
        self.vel_pub = rospy.Publisher(
            self.config.target.move_vel, Twist, queue_size=1
        )
        self.curr_goal = BasePose()
        rospy.Subscriber(
            self.config.target.move_vel, Twist,
            self._move_vel_callback)
        self.localizer = self.get_localize_algorithm_from_config()

    def _move_vel_callback(self, data):
        if data is None:
            return
        self.curr_vel = BasePose(
            x=data.linear.x,
            y=data.linear.y,
        )

    def get_localize_algorithm_from_config(self, **param):
        localizer = None
        _param = dict()
        if self.config.localizer and self.config.localizer.algorithm:
            try:
                _ = import_module(f"robosdk.algorithms.localize")
                localizer = ClassFactory.get_cls(
                    ClassType.LOCALIZE, self.config.localizer.algorithm)
            except Exception as e:
                self.logger.error(
                    f"Fail to initial localizer algorithm "
                    f"{self.config.localizer.algorithm} : {e}")
        if not localizer:
            return
        if (self.config.localizer and
                isinstance(self.config.localizer.parameter, list)):
            _param = {
                p["key"]: p.get("value", "")
                for p in self.config.localizer.parameter
                if isinstance(p, dict) and "key" in p
            }
        _param.update(**param)
        return localizer(**_param)

    def _transform_goal(self, goal: BasePose):
        import tf.transformations
        from geometry_msgs.msg import PoseStamped
        from geometry_msgs.msg import Pose
        from geometry_msgs.msg import Point
        from geometry_msgs.msg import Quaternion
        from move_base_msgs.msg import MoveBaseGoal

        self.goal_lock.acquire()
        q = tf.transformations.quaternion_from_euler(0, 0, goal.z)
        pose_stamped = PoseStamped(
            pose=Pose(
                Point(goal.x, goal.y, 0.000),
                Quaternion(q[0], q[1], q[2], q[3])
            )
        )
        pose_stamped.header.frame_id = self.config.target.mapframe
        target = MoveBaseGoal()
        target.target_pose = pose_stamped
        self.curr_goal = goal.copy()
        self.goal_lock.release()
        return target

    def cancel_goal(self):
        from actionlib_msgs.msg import GoalID

        self.logger.warning("Goal Cancel")
        try:
            self.move_base_cancel_goal_pub.publish(GoalID())
        except Exception as err:
            self.logger.debug(f"Cancel goal failure: {err}")
            self.move_base_ac.cancel_all_goals()
        self.stop()

    def _move_base_status_callback(self, msg):
        if msg.status_list:
            self.execution_status = getattr(msg.status_list[-1], "status")

    def _laser_scan_callback(self, msg):
        self.scan_msg = msg

    def execute_track(self, plan: PathNode, min_gap: float = 0):
        target = plan.copy()
        if not min_gap:
            min_gap = self.config.limited.min_distance
        while 1:
            curr_point = self.get_location()
            if curr_point - target <= abs(min_gap):
                target = plan.next
            if target is None:
                self.logger.info("Path planning execute complete")
                break
            self.goto(goal=target.position)

    def get_location(self, **parameter):
        if self.localizer and hasattr(self.localizer, "get_curr_state"):
            return self.localizer.get_curr_state(**parameter)
        return self.curr_goal.copy()

    def goto(self, goal: BasePose,
             start_pos: BasePose = None,
             limit_time: int = 0):  # noqa
        if start_pos is None:
            start_pos = self.get_location()
        curr_goal = self._get_absolute_pose(goal, start_pos)
        return self.goto_absolute(curr_goal, limit_time=limit_time)

    def goto_absolute(self, goal: BasePose, limit_time: int = 0):
        self.logger.info(f"Sending the goal, {str(goal)}")
        target = self._transform_goal(goal)
        return self._send_action_goal(target, limit_time=limit_time)

    @staticmethod
    def _get_absolute_pose(goal: BasePose, base: BasePose):
        nx = base.x + goal.x * np.cos(base.z) - goal.y * np.sin(base.z)
        ny = base.y + goal.x * np.sin(base.z) + goal.y * np.cos(base.z)
        nz = base.z + goal.z
        return BasePose(x=nx, y=ny, z=nz)

    def _send_action_goal(self, goal, limit_time: int = 0):
        import rospy

        self.logger.debug("Waiting for the server")
        self.move_base_ac.wait_for_server()

        self.move_base_ac.send_goal(goal)

        self.logger.debug("Waiting for the move_base Result")
        exit_without_time = True
        if not limit_time:
            limit_time = int(self.config.limited.exec_time)
        if limit_time > 0:
            exit_without_time = self.move_base_ac.wait_for_result(
                rospy.Duration(limit_time)
            )
        while 1:
            if not exit_without_time:
                self.cancel_goal()
                self.logger.warning("ERROR:Timed out achieving goal")
                return False
            if ((self.move_base_as and self.move_base_as.should_stop) or
                    (self.execution_status == ActionStatus.ABORTED.value)):
                self.cancel_goal()
                return False
            if (isinstance(self.move_base_as, RoboActionHandle)
                    and self.move_base_as.is_preempt):
                self.cancel_goal()
                return False
            if self.execution_status == ActionStatus.SUCCEEDED.value:
                return True
            rospy.sleep(0.1)

    def set_vel(self,
                forward: float = 0,
                turn: float = 0,
                execution: int = 1):
        """
        set velocity to robot
       :param forward: linear velocity in m/s
       :param turn: rotational velocity in m/s
       :param execution: execution time in seconds
       """

        import rospy
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x = forward
        msg.angular.z = turn

        stop_time = rospy.get_rostime() + rospy.Duration(execution)
        while rospy.get_rostime() < stop_time:
            self.vel_pub.publish(msg)
            rospy.sleep(.05)

    def turn_right(self):
        self.set_vel(1.5, -.7)

    def turn_left(self):
        self.set_vel(1.5, .7)

    def go_forward(self):
        self.set_vel(1.3, 0)

    def go_backward(self):
        self.set_vel(-.5, 0)

    def stop(self):
        self.set_vel(0, 0)

    def go_forward_util(self, stop_distance: float = 0.5):
        while self.get_front_distance() > stop_distance:
            self.set_vel(.1, 0)
        self.stop()

    def speed(self, linear: float = 0., rotational: float = 0.):
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = rotational

        self.vel_pub.publish(msg)

    def get_front_distance(self, degree: int = 10) -> float:
        if self.scan_msg is not None:
            all_data = getattr(self.scan_msg, "ranges", [])
            if not len(all_data):
                return 0.0
            circle = int(len(all_data) / 2)
            left_degree = max(circle - abs(degree), 10)
            right_degree = min(circle + abs(degree), len(all_data))
            ranges = list(
                filter(lambda x: (x != float('inf') and x != float("-inf")),
                       all_data[left_degree:right_degree])
            )
            return float(np.mean(ranges))
        return 0.0

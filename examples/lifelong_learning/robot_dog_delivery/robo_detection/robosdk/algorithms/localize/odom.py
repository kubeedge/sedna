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
import copy

from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory
from robosdk.utils.schema import BasePose

from .base import Localize


@ClassFactory.register(ClassType.LOCALIZE, alias="odom")
class Odom(Localize):  # noqa
    def __init__(self, name: str = "Odometry",
                 mapframe: str = "map",
                 topic: str = "/odom",
                 pose_pub: str = "/initialpose"
                 ):
        super(Odom, self).__init__(name=name)

        import rospy
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseWithCovarianceStamped

        rospy.Subscriber(
            topic,
            Odometry,
            self.get_odom_state,
        )
        self.initial_pose = rospy.Publisher(
            pose_pub,
            PoseWithCovarianceStamped,
            queue_size=1
        )
        self.mapframe = mapframe or "map"

    def get_odom_state(self, msg):

        import tf.transformations

        self.state_lock.acquire()
        orientation = msg.pose.pose.orientation
        _, _, z = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        self.curr_state.x = msg.pose.pose.position.x
        self.curr_state.y = msg.pose.pose.position.y
        self.curr_state.z = z
        self.state_lock.release()

    def get_curr_state(self, **kwargs) -> BasePose:
        state = copy.deepcopy(self.curr_state)
        return state

    def set_curr_state(self, state: BasePose):

        import rospy
        import tf.transformations
        from geometry_msgs.msg import PoseWithCovarianceStamped

        self.curr_state.x = state.x
        self.curr_state.y = state.y
        self.curr_state.z = state.z
        self.curr_state.w = state.w

        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.seq = 1
        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.header.frame_id = self.mapframe

        q = tf.transformations.quaternion_from_euler(0, 0, state.z)
        initial_pose.pose.pose.position.x = state.x
        initial_pose.pose.pose.position.y = state.y
        initial_pose.pose.pose.position.z = 0.0

        initial_pose.pose.pose.orientation.x = q[0]
        initial_pose.pose.pose.orientation.y = q[1]
        initial_pose.pose.pose.orientation.z = q[2]
        initial_pose.pose.pose.orientation.w = q[3]
        for _ in range(10):
            self.initial_pose.publish(initial_pose)
            rospy.sleep(0.1)

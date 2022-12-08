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
import threading

from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory
from robosdk.utils.schema import TaskNode
from robosdk.utils.constant import ActionStatus

from .base import TaskAgent
from .base import RoboActionHandle


__all__ = ("RosActionService", )


@ClassFactory.register(ClassType.GENERAL)
class RosActionService(TaskAgent):  # noqa

    def __init__(self, name: str, max_task: int = 100):
        super(RosActionService, self).__init__(name=name, max_task=max_task)

        import rospy

        action_namespace = rospy.get_namespace()
        self.namespace = f"{action_namespace}{self.agent_name}"
        self._all_url = []

    def stop(self):
        return map(lambda x: x.shutdown(), self._all_url)

    def start(self):
        import rospy

        from robosdk.msgs.cloud_msgs.srv import RobotActions    # noqa

        self._all_url = [
            rospy.Service(
                f"{self.namespace}/list",
                RobotActions,
                self.get_all_task
            ),
            rospy.Service(
                f"{self.namespace}/create",
                RobotActions,
                self.create_task
            ),
            rospy.Service(
                f"{self.namespace}/state",
                RobotActions,
                self.get_task_state
            ),
            rospy.Service(
                f"{self.namespace}/delete",
                RobotActions,
                self.delete_task
            )
        ]

    def create_task(self, request):  # noqa
        robo_id = request.robot_id
        task_id = request.task_id
        task_name = request.name
        task_instance = request.instance
        task_type = request.task_type
        self.seq += 1
        task = TaskNode(
            taskId=task_id,
            robotId=robo_id,
            name=task_name,
            instance=task_instance,
            taskType=task_type
        )
        self.all_task[task_id] = {
            "seq": self.seq,
            "task": task,
            "state": ActionStatus.PENDING.value
        }
        th = RoboActionHandle(seq=self.seq, task=task)
        self._add(th)
        return self.all_task[task_id]

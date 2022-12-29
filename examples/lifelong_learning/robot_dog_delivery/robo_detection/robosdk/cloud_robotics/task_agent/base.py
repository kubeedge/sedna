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
import abc
import queue
import copy
import asyncio
import threading
from typing import Dict

from robosdk.utils.logger import logging
from robosdk.utils.constant import ActionStatus
from robosdk.utils.schema import TaskNode


class RoboActionHandle:

    def __init__(self, seq: int = 0, task: TaskNode = None):
        self.action_lock = threading.RLock()
        self.state = ActionStatus.PENDING
        self.task = task
        self.seq = seq

    def get_state(self):
        self.action_lock.acquire()
        state = copy.copy(self.state)
        self.action_lock.release()
        return state

    def set_state(self, state):
        self.action_lock.acquire()
        self.state = state
        self.action_lock.release()

    @property
    def is_preempt(self):
        return self.get_state() in (
            ActionStatus.PREEMPTING,
            ActionStatus.RECALLED
        )

    @property
    def is_available(self):
        return self.get_state() not in (
            ActionStatus.ACTIVE,
            ActionStatus.PREEMPTING,
            ActionStatus.RECALLED
        )


class TaskAgent(metaclass=abc.ABCMeta):

    def __init__(self, name: str = "task_agent", max_task: int = 100):
        self.seq = 0
        self.agent_name = name
        self.logger = logging.bind(instance="taskAgent", system=True)
        self.all_task = {}
        self.mq = queue.Queue(maxsize=max_task)
        self.executors = {}

    def initial_executors(self, executors: Dict):
        self.executors = executors

    @abc.abstractmethod
    def start(self):
        ...

    @abc.abstractmethod
    def stop(self):
        ...

    def get_all_task(self):
        return self.all_task

    def create_task(self):
        pass

    def get_task_state(self):
        pass

    def delete_task(self):
        pass

    def _get(self):
        try:
            task_id = self.mq.get()
        except queue.Empty:
            return None
        else:
            return self.all_task.get(task_id, None)

    def _add(self, task_id: RoboActionHandle):
        try:
            self.mq.put_nowait(task_id)
        except queue.Full:
            self.logger.warning('mq full, drop the message')
        except Exception as e:
            self.logger.error(f'mq add message fail: {e}')

    def _run(self):
        while 1:
            data = self._get()
            if not isinstance(data, RoboActionHandle):
                continue
            robot = self.executors.get(data.task.robotId, None)
            if not getattr(robot, "has_connect", False):
                self.logger.warning(f"Robot {robot} has not been initialized")
                continue
            task_type = data.task.taskType.lower()
            task_instance = data.task.instance
            parameter = data.task.parameter
            loop = asyncio.get_event_loop()
            self.logger.info(
                f"Staring running task {data.seq}: {data.task.taskId}")
            try:
                robot.navigation.set_action_server(data)
                driver = getattr(getattr(robot, task_type), task_instance)
                loop.run_until_complete(
                    driver(**parameter)
                )
            except Exception as err:
                data.set_state(ActionStatus.ABORTED)
                self.logger.error(err)
                break
            else:
                data.set_state(ActionStatus.SUCCEEDED)
                self.logger.info(
                    f"Complete task {data.seq}: {data.task.taskId}")

    def run(self):
        mq = threading.Thread(target=self._run)
        mq.setDaemon(True)
        mq.start()

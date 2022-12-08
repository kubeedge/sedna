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
from enum import Enum


class PgmItem(Enum):
    UNKNOWN = -1
    FREE = 0
    OBSTACLE = 100


class MsgChannelItem(Enum):
    CONNECT_INTERVAL = 5
    GET_MESSAGE_TIMEOUT = 10


class ActionStatus(Enum):
    PENDING = 0
    ACTIVE = 1
    PREEMPTED = 2
    SUCCEEDED = 3
    ABORTED = 4
    REJECTED = 5
    PREEMPTING = 6
    RECALLING = 7
    RECALLED = 8
    LOST = 9
    UNKONWN = 99


class GaitType(Enum):
    RUN = 3
    TROT = 0
    FALL = 2
    UPSTAIR = 1
    LIEON = 11
    UNKONWN = 99

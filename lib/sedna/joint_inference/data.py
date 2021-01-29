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

import json


class ServiceInfo:
    def __init__(self):
        self.startTime = ''
        self.updateTime = ''
        self.inferenceNumber = 0
        self.hardExampleNumber = 0
        self.uploadCloudRatio = 0

    @staticmethod
    def from_json(json_str):
        info = ServiceInfo()
        info.__dict__ = json.loads(json_str)
        return info

    def to_json(self):
        info = json.dumps(self.__dict__)
        return info

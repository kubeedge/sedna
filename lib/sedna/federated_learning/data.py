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
import logging

from sedna.common.utils import obj_to_pickle_string, pickle_string_to_obj

LOG = logging.getLogger(__name__)


class AggregationData:
    """Struct of aggregation data."""

    def __init__(self):
        self.round_number = 0
        self.location_id = ""
        self.task_id = ""
        self.worker_id = ""
        self.flatten_weights = None
        self.shapes = None
        self.sample_count = 0
        self.exit_flag = False  # if true, exit federated training.
        self.ip_port = ""
        self.start_time = ""
        self.task_info = None  # federated task info, json format.

    @staticmethod
    def from_json(json_str):
        ad = AggregationData()
        ad.__dict__ = json.loads(json_str)
        ad.flatten_weights = pickle_string_to_obj(ad.flatten_weights)
        return ad

    def to_json(self):
        self.flatten_weights = obj_to_pickle_string(self.flatten_weights)
        s = json.dumps(self.__dict__)
        return s


class JobInfo:
    def __init__(self):
        self.startTime = ""
        self.updateTime = ""
        self.currentRound = 0
        self.sampleCount = 0

    def to_json(self):
        s = json.dumps(self.__dict__)
        return s

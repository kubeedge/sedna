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

"""Aggregation algorithms"""

import abc


class BaseAggregation(metaclass=abc.ABCMeta):
    def __init__(self):
        self.total_size = 0
        self.weights = None

    @abc.abstractmethod
    def aggregate(self, weights, size=0):
        """
        Aggregation
        :param weights: deep learning weight
        :param size: numbers of sample in each loop
        """

    @abc.abstractmethod
    def exit_check(self):
        """check if should exit federated learning job"""

    @abc.abstractmethod
    def client_choose(self):
        """choosing client to join federated learning in this round"""

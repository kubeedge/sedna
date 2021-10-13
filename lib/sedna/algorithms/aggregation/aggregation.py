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
from copy import deepcopy
from typing import List

from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('AggClient', 'FedAvg',)


class AggClient:
    """
    Client that interacts with cloud aggregator

    Parameters
    ----------
    num_samples: int
        number of samples for the current weights
    weights: List
        weights of the layer as a list of number-like array,
        such as [[0, 0, 0, 0], [0, 0, 0, 0] ... ]
    """

    num_samples: int
    weights: List


class BaseAggregation(metaclass=abc.ABCMeta):
    """
    Abstract class of aggregator
    """

    def __init__(self):
        self.total_size = 0
        self.weights = None

    @abc.abstractmethod
    def aggregate(self, clients: List[AggClient]):
        """
        Some algorithms can be aggregated in sequence,
        but some can be calculated only after all aggregated data is uploaded.
        therefore, this abstractmethod should consider that all weights are
        uploaded.

        Parameters
        ----------
        clients: List
            All clients in federated learning job

        Returns
        -------
        Array-like
            final weights use to update model layer
        """


@ClassFactory.register(ClassType.FL_AGG)
class FedAvg(BaseAggregation, abc.ABC):
    """
    Federated averaging algorithm
    """

    def aggregate(self, clients: List[AggClient]):
        """
        Calculate the average weight according to the number of samples

        Parameters
        ----------
        clients: List
            All clients in federated learning job

        Returns
        -------
        update_weights : Array-like
            final weights use to update model layer
        """
        import numpy as np

        if not len(clients):
            return self.weights
        self.total_size = sum([c.num_samples for c in clients])
        old_weight = [np.zeros(np.array(c).shape) for c in
                      next(iter(clients)).weights]
        updates = []
        for inx, row in enumerate(old_weight):
            for c in clients:
                row += (np.array(c.weights[inx]) * c.num_samples
                        / self.total_size)
            updates.append(row.tolist())
        self.weights = deepcopy(updates)
        return updates


@ClassFactory.register(ClassType.FL_AGG)
class MistNet(BaseAggregation, abc.ABC):
    def __init__(self, cut_layer, epsilon=100):
        super().__init__()
        self.parameters = {
            "type": "mistnet",
            "cut_layer": cut_layer,
            "epsilon": epsilon
        }
        if isinstance(self.parameters["cut_layer"], str):
            if self.parameters["cut_layer"].isdigit():
                self.parameters["cut_layer"] = int(cut_layer)

        if isinstance(self.parameters["epsilon"], str):
            if self.parameters["epsilon"].isdigit():
                self.parameters["epsilon"] = int(cut_layer)

    def aggregate(self, clients: List[AggClient]):
        pass


@ClassFactory.register(ClassType.FL_AGG)
class FedAvgV2(BaseAggregation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "type": "fedavg",
        }

    def aggregate(self, clients: List[AggClient]):
        pass

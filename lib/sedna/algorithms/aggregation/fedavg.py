import abc
from copy import deepcopy

import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType
from .base import BaseAggregation

__all__ = ('FedAvg',)


@ClassFactory.register(ClassType.FL_AGG)
class FedAvg(BaseAggregation, abc.ABC):
    """Federated averaging algorithm"""

    def aggregate(self, weights, size=0):
        total_sample = self.total_size + size
        if not total_sample:
            return self.weights
        updates = []
        for inx, weight in enumerate(weights):
            old_weight = self.weights[inx]
            row_weight = ((np.array(weight) - old_weight) *
                          (size / total_sample) + old_weight)
            updates.append(row_weight)
        self.weights = deepcopy(updates)
        return updates

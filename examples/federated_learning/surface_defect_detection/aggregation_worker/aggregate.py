import logging

import numpy as np

from sedna.federated_learning.aggregator import AggregationServer
from sedna.federated_learning.aggregator import Aggregator

LOG = logging.getLogger(__name__)


class FooAggregator(Aggregator):

    def __init__(self):
        super().__init__()

    def aggregate(self):
        LOG.info("start aggregate in FooAgg")
        self.agg_data_dict_aggregated = self.agg_data_dict
        self.agg_data_dict = {}

        for k, v in self.agg_data_dict_aggregated.items():
            LOG.info(f"ip in aggregated={v.ip_port}")


class FedAvgAggregator(Aggregator):
    def __init__(self):
        super().__init__()

    def aggregate(self):
        LOG.info("start aggregate in FedAvgAggregator")
        new_weights = [
            np.zeros(a.shape) for a
            in next(iter(self.agg_data_dict.values())).flatten_weights]
        total_size = sum([a.sample_count for a in self.agg_data_dict.values()])

        for c in self.agg_data_dict.values():
            for i in range(len(c.flatten_weights)):
                old_weights = (
                        np.array(
                            c.flatten_weights[i]) * c.sample_count / total_size
                )
                new_weights[i] += old_weights

        self.agg_data_dict_aggregated = self.agg_data_dict
        self.agg_data_dict = {}

        for _, d in self.agg_data_dict_aggregated.items():
            d.flatten_weights = new_weights


if __name__ == '__main__':
    agg = AggregationServer(FedAvgAggregator())

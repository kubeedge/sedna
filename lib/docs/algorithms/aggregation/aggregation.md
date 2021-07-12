Module sedna.algorithms.aggregation.aggregation
===============================================
Aggregation algorithms

Classes
-------

`FedAvg()`
:   Federated averaging algorithm : Calculate the average weight
    according to the number of samples

    ### Ancestors (in MRO)

    * sedna.algorithms.aggregation.aggregation.BaseAggregation
    * abc.ABC

    ### Methods

    `aggregate(self, weights, size=0)`
    :   Some algorithms can be aggregated in sequence,
        but some can be calculated only after all aggregated data is uploaded.
        therefore, this abstractmethod should consider that all weights are
        uploaded.
        :param weights: weights received from node
        :param size: numbers of sample in each loop
        :return: final weights
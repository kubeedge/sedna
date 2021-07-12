Module sedna.core.federated_learning.federated_learning
=======================================================

Classes
-------

`FederatedLearning(estimator, aggregation='FedAvg')`
:   Federated learning
    
    Initial a FederatedLearning job
    :param estimator: Customize estimator
    :param aggregation: aggregation algorithm for FederatedLearning

    ### Ancestors (in MRO)

    * sedna.core.base.JobBase

    ### Methods

    `register(self, timeout=300)`
    :

    `train(self, train_data, valid_data=None, post_process=None, **kwargs)`
    :   Training task for FederatedLearning
        :param train_data: datasource use for train
        :param valid_data: datasource use for evaluation
        :param post_process: post process
        :param kwargs: params for training of customize estimator
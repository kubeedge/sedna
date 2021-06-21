Module sedna.algorithms.multi_task_learning.multi_task_learning
===============================================================

Classes
-------

`MulTaskLearning(estimator=None, task_definition=None, task_relationship_discovery=None, task_mining=None, task_remodeling=None, inference_integrate=None)`
:   

    ### Static methods

    `parse_param(param_str)`
    :

    ### Methods

    `evaluate(self, data: sedna.datasources.BaseDataSource, metrics=None, metrics_param=None, **kwargs)`
    :

    `predict(self, data: sedna.datasources.BaseDataSource, post_process=None, **kwargs)`
    :

    `train(self, train_data: sedna.datasources.BaseDataSource, valid_data: sedna.datasources.BaseDataSource = None, post_process=None, **kwargs)`
    :
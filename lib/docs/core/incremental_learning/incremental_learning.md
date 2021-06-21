Module sedna.core.incremental_learning.incremental_learning
===========================================================

Classes
-------

`IncrementalLearning(estimator)`
:   Incremental learning
    
    Initial a IncrementalLearning job
    :param estimator: Customize estimator

    ### Ancestors (in MRO)

    * sedna.core.base.JobBase

    ### Methods

    `evaluate(self, data, post_process=None, **kwargs)`
    :   Evaluate task for IncrementalLearning
        :param data: datasource use for evaluation
        :param post_process: post process
        :param kwargs: params for evaluate of customize estimator
        :return: evaluate metrics

    `inference(self, data=None, post_process=None, **kwargs)`
    :   Inference task for IncrementalLearning
        :param data: inference sample
        :param post_process: post process
        :param kwargs: params for inference of customize estimator
        :return: inference result, result after post_process, if is hard sample

    `train(self, train_data, valid_data=None, post_process=None, **kwargs)`
    :   Training task for IncrementalLearning
        :param train_data: datasource use for train
        :param valid_data: datasource use for evaluation
        :param post_process: post process
        :param kwargs: params for training of customize estimator
        :return: estimator
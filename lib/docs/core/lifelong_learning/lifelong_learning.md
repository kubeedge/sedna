Module sedna.core.lifelong_learning.lifelong_learning
=====================================================

Classes
-------

`LifelongLearning(estimator, task_definition=None, task_relationship_discovery=None, task_mining=None, task_remodeling=None, inference_integrate=None, unseen_task_detect=None)`
:   Lifelong learning
    
    Initial a lifelong learning job
    :param estimator: Customize estimator
    :param task_definition: dict, {"method": "", param: ""} Multitask definition base on given traning samples
    :param task_relationship_discovery: dict, {"method": "", param: ""}  Discover the relation of tasks which generated from task_definition  # noqa
    :param task_mining:  dict, {"method": "", param: ""} Mining target tasks of inference samples
    :param task_remodeling:  dict, {"method": "", param: ""} Remodeling tasks
    :param inference_integrate:  dict, {"method": "", param: ""} Integrating algorithm for the output geted by multitask inference  # noqa
    :param unseen_task_detect:  dict, {"method": "", param: ""} unseen task detect

    ### Ancestors (in MRO)

    * sedna.core.base.JobBase

    ### Methods

    `evaluate(self, data, post_process=None, **kwargs)`
    :   Evaluate task for LifelongLearning
        :param data: datasource use for evaluation
        :param post_process: post process
        :param kwargs: params for evaluate of customize estimator
        :return: evaluate metrics

    `inference(self, data=None, post_process=None, **kwargs)`
    :   Inference task for LifelongLearning
        :param data: inference sample
        :param post_process: post process
        :param kwargs: params for inference of customize estimator
        :return: inference result, if is hard sample, match tasks

    `train(self, train_data, valid_data=None, post_process=None, action='initial', **kwargs)`
    :   :param train_data: data use to train model
        :param valid_data: data use to valid model
        :param post_process: callback function
        :param action: initial - kb init, update - kb incremental update

    `update(self, train_data, valid_data=None, post_process=None, **kwargs)`
    :
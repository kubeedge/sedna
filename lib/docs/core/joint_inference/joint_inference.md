Module sedna.core.joint_inference.joint_inference
=================================================

Classes
-------

`JointInference(estimator=None)`
:   Joint inference
    
    Initial a JointInference Job
    :param estimator: Customize estimator

    ### Ancestors (in MRO)

    * sedna.core.base.JobBase

    ### Methods

    `inference(self, data=None, post_process=None, **kwargs)`
    :   Inference task for IncrementalLearning
        :param data: inference sample
        :param post_process: post process
        :param kwargs: params for inference of customize estimator
        :return: if is hard sample, real result,
        little model result, big model result

    `train(self, train_data, valid_data=None, post_process=None, **kwargs)`
    :   todo: no support yet

`TSBigModelService(estimator=None)`
:   Large model services implemented
    Provides RESTful interfaces for large-model inference.
    
    Initial a big model service for JointInference
    :param estimator: Customize estimator

    ### Ancestors (in MRO)

    * sedna.core.base.JobBase

    ### Methods

    `inference(self, data=None, post_process=None, **kwargs)`
    :   Inference task for IncrementalLearning
        :param data: inference sample
        :param post_process: post process
        :param kwargs: params for inference of big model
        :return: inference result

    `start(self)`
    :   Start inference rest server
        :return:

    `train(self, train_data, valid_data=None, post_process=None, **kwargs)`
    :   todo: no support yet
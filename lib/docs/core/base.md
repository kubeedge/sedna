Module sedna.core.base
======================

Classes
-------

`JobBase(estimator, config=None)`
:   sedna feature base class

    ### Descendants

    * sedna.core.federated_learning.federated_learning.FederatedLearning
    * sedna.core.incremental_learning.incremental_learning.IncrementalLearning
    * sedna.core.joint_inference.joint_inference.JointInference
    * sedna.core.joint_inference.joint_inference.TSBigModelService
    * sedna.core.lifelong_learning.lifelong_learning.LifelongLearning

    ### Class variables

    `parameters`
    :   The Context provides the capability of obtaining the context

    ### Instance variables

    `initial_hem`
    :   initial hard_example_mining_algorithms instance
        env:
            `HEM_NAME`: string, hard_example_mining_algorithms name
            `HEM_PARAMETERS`: json_str, parameters of hem

    `model_path`
    :   :return: model save/load path

    ### Methods

    `evaluate(self, data, post_process=None, **kwargs)`
    :   Evaluate the model based on test data
        :param data: eval data sources
        :param post_process: post process, string
        :param kwargs: parameters for evaluate
        :return:

    `get_parameters(self, param, default=None)`
    :   Get parameters from the environment context
        :param param: key in env
        :param default: return if value is None
        :return:

    `inference(self, x=None, post_process=None, **kwargs)`
    :   Use the model to predict the result
        :param x: input_sample
        :param post_process: post process, string
        :param kwargs: parameters for inference
        :return:

    `report_task_info(self, task_info, status, results, kind='train')`
    :   Send task info to lc client

    `train(self, **kwargs)`
    :   Generate model from training data and estimator
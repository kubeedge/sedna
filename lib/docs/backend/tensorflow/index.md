Module sedna.backend.tensorflow
===============================

Classes
-------

`KerasBackend(estimator, fine_tune=True, **kwargs)`
:   ML Framework Backend base Class

    ### Ancestors (in MRO)

    * sedna.backend.tensorflow.TFBackend
    * sedna.backend.base.BackendBase

    ### Methods

    `set_session(self)`
    :

`TFBackend(estimator, fine_tune=True, **kwargs)`
:   ML Framework Backend base Class

    ### Ancestors (in MRO)

    * sedna.backend.base.BackendBase

    ### Descendants

    * sedna.backend.tensorflow.KerasBackend

    ### Methods

    `finetune(self)`
    :   todo: no support yet

    `get_weights(self)`
    :   todo: no support yet

    `load_weights(self)`
    :

    `model_info(self, model, relpath=None, result=None)`
    :

    `set_weights(self, weights)`
    :   todo: no support yet
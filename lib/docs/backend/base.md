Module sedna.backend.base
=========================

Classes
-------

`BackendBase(estimator, fine_tune=True, **kwargs)`
:   ML Framework Backend base Class

    ### Descendants

    * sedna.backend.tensorflow.TFBackend

    ### Static methods

    `parse_kwargs(func, **kwargs)`
    :

    ### Instance variables

    `model_name`
    :

    ### Methods

    `evaluate(self, **kwargs)`
    :   evaluate model.

    `get_weights(self)`
    :   Get the weights.

    `load(self, model_url='', model_name=None, **kwargs)`
    :

    `model_info(self, model, relpath=None, result=None)`
    :

    `predict(self, **kwargs)`
    :   Inference model.

    `predict_proba(self, **kwargs)`
    :   Compute probabilities of possible outcomes for samples in X.

    `save(self, model_url='', model_name=None)`
    :

    `set_weights(self, weights)`
    :   Set weight with memory tensor.

    `train(self, **kwargs)`
    :   Train model.
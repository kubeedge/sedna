import os
import traceback

import torch
from torch.backends import cudnn

from sedna.backend.base import BackendBase
from sedna.common.log import LOGGER

class TorchBackend(BackendBase):
    def __init__(self, estimator, fine_tune=True, **kwargs):
        super(TorchBackend, self).__init__(
            estimator=estimator, fine_tune=fine_tune, **kwargs)
        self.framework = "pytorch"
        self.device = "cuda" if torch.cuda.is_available() and self.use_cuda else "cpu"
        cudnn.benchmark = True

        if callable(self.estimator):
            self.estimator = self.estimator()

    def evaluate(self, **kwargs):
        if not self.has_load:
            self.load()
        return self.estimator.evaluate(**kwargs)

    def train(self, **kwargs):
        """ Not implemented!"""
        pass

    def predict(self, data, **kwargs):
        if not self.has_load:
            self.load()
        return self.estimator.predict(data=data, **kwargs)

    def load(self, model_url="", model_name=None, **kwargs):
        model_path = self.model_save_path
        if os.path.exists(model_path):
            try:
                self.estimator.load(model_path, **kwargs)
            except Exception as e:
                LOGGER.error(f"Failed to load the model - {e}")
                LOGGER.error(traceback.format_exc())
                self.has_load = False
        else:
            LOGGER.info("Path to model does not exists!")

        self.has_load = True
import os

import torch

from nn import *
from sedna.backend.base import BackendBase
from sedna.common.file_ops import FileOps

class TorchBackend(BackendBase):
    def __init__(self, estimator, fine_tune=True, **kwargs):
        super(TorchBackend, self).__init__(
            estimator=estimator, fine_tune=fine_tune, **kwargs)
        self.framework = "pytorch"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if callable(self.estimator):
            self.estimator = self.estimator()

    def evaluate(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def predict(self, data, **kwargs):
        return self.estimator.predict(data=data, **kwargs)

    def load(self, model_url, model_name, **kwargs):
        model_path = FileOps.join_path(self.model_save_path, self.model_name)
        if os.path.exists(model_path):
            return self.estimator.load(model_path)


    def load_weights(self):
        model_path = FileOps.join_path(self.model_save_path, self.model_name)
        if os.path.exists(model_path):
            self.estimator.load_weights(model_path)



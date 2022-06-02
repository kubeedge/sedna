# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        self.has_load = False

        self.device = "cpu"
        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = "cuda"
        cudnn.benchmark = False

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
                self.estimator.load(**kwargs)
            except Exception as e:
                LOGGER.error(f"Failed to load the model - {e}")
                LOGGER.error(traceback.format_exc())
                self.has_load = False
        else:
            LOGGER.info("Path to model does not exists!")

        self.has_load = True

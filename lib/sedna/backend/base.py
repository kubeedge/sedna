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

"""ML Framework Backend base Class"""
from copy import deepcopy
from sedna.common.file_ops import FileOps


class BackendBase:
    """ML Framework Backend base Class"""

    def __init__(self, estimator, fine_tune=True, **kwargs):
        self.framework = ""
        self.estimator = estimator
        self.use_cuda = True if kwargs.get("use_cuda") else False
        self.fine_tune = fine_tune
        self.model_save_path = kwargs.get("model_save_path") or "/tmp"
        self.has_load = False

    @property
    def model_name(self):
        model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".pb"}
        continue_flag = "_finetune_" if self.fine_tune else ""
        post_fix = model_postfix.get(self.framework, ".pkl")
        return f"model{continue_flag}{self.framework}{post_fix}"

    def train(self, **kwargs):
        """Train model."""
        if callable(self.estimator):
            self.estimator = self.estimator()
        return self.estimator.train(**kwargs)

    def predict(self, **kwargs):
        """Inference model."""
        return self.estimator.predict(**kwargs)

    def predict_proba(self, **kwargs):
        """Compute probabilities of possible outcomes for samples in X."""
        return self.estimator.predict_proba(**kwargs)

    def evaluate(self, **kwargs):
        """evaluate model."""
        return self.estimator.evaluate(**kwargs)

    def save(self, model_url="", model_name=None):
        mname = model_name or self.model_name
        model_path = FileOps.join_path(self.model_save_path, mname)
        self.estimator.save(model_path)
        if model_url and FileOps.exists(model_path):
            FileOps.upload(model_path, model_url)
        return model_path

    def load(self, model_url="", model_name=None):
        mname = model_name or self.model_name
        model_path = FileOps.join_path(self.model_save_path, mname)
        if model_url:
            FileOps.download(model_url, model_path)
        self.has_load = True

        return self.estimator.load(model_path)

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        raise NotImplementedError

    def get_weights(self):
        """Get the weights."""
        raise NotImplementedError

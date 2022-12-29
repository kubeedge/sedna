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

import os.path
from inspect import getfullargspec

from sedna.common.file_ops import FileOps


class BackendBase:
    """ML Framework Backend base Class"""

    def __init__(self, estimator, fine_tune=True, **kwargs):
        self.framework = ""
        self.estimator = estimator
        self.use_cuda = True if kwargs.get("use_cuda") else False
        self.use_npu = True if kwargs.get("use_npu") else False
        self.fine_tune = fine_tune
        self.model_save_path = kwargs.get("model_save_path") or "/tmp"
        self.default_name = kwargs.get("model_name")
        self.has_load = False

    @property
    def model_name(self):
        if self.default_name:
            return self.default_name
        model_postfix = {"pytorch": [".pth", ".pt"],
                         "keras": ".pb",
                         "tensorflow": ".pb",
                         "mindspore": ".ckpt"}
        continue_flag = "_finetune_" if self.fine_tune else ""
        post_fix = model_postfix.get(self.framework, ".pkl")
        return f"model{continue_flag}{self.framework}{post_fix}"

    @staticmethod
    def parse_kwargs(func, **kwargs):
        if not callable(func):
            return kwargs
        need_kw = getfullargspec(func)
        if need_kw.varkw == 'kwargs':
            return kwargs
        return {k: v for k, v in kwargs.items() if k in need_kw.args}

    def train(self, *args, **kwargs):
        """Train model."""
        if callable(self.estimator):
            varkw = self.parse_kwargs(self.estimator, **kwargs)
            self.estimator = self.estimator(**varkw)
        fit_method = getattr(self.estimator, "fit", self.estimator.train)
        varkw = self.parse_kwargs(fit_method, **kwargs)
        return fit_method(*args, **varkw)

    def update(self, *args, **kwargs):
        """Update model."""
        if callable(self.estimator):
            varkw = self.parse_kwargs(self.estimator, **kwargs)
            self.estimator = self.estimator(**varkw)
        fit_method = getattr(self.estimator, "fit", self.estimator.update)
        varkw = self.parse_kwargs(fit_method, **kwargs)
        return fit_method(*args, **varkw)

    def predict(self, *args, **kwargs):
        """Inference model."""
        varkw = self.parse_kwargs(self.estimator.predict, **kwargs)
        return self.estimator.predict(*args, **varkw)

    def predict_proba(self, *args, **kwargs):
        """Compute probabilities of possible outcomes for samples in X."""
        varkw = self.parse_kwargs(self.estimator.predict_proba, **kwargs)
        return self.estimator.predict_proba(*args, **varkw)

    def evaluate(self, *args, **kwargs):
        """evaluate model."""
        varkw = self.parse_kwargs(self.estimator.evaluate, **kwargs)
        return self.estimator.evaluate(*args, **varkw)

    def save(self, model_url="", model_name=None):
        mname = model_name or self.model_name
        if os.path.isfile(self.model_save_path):
            self.model_save_path, mname = os.path.split(self.model_save_path)

        FileOps.clean_folder([self.model_save_path], clean=False)
        model_path = FileOps.join_path(self.model_save_path, mname)
        self.estimator.save(model_path)
        if model_url and FileOps.exists(model_path):
            FileOps.upload(model_path, model_url)
            model_path = model_url
        return model_path

    def model_info(self, model, relpath=None, result=None):
        _, _type = os.path.splitext(model)
        if relpath:
            _url = FileOps.remove_path_prefix(model, relpath)
        else:
            _url = model
        results = [{
            "format": _type.lstrip("."),
            "url": _url,
            "metrics": result
        }]
        return results

    def load(self, model_url="", model_name=None, **kwargs):
        mname = model_name or self.model_name
        if callable(self.estimator):
            varkw = self.parse_kwargs(self.estimator, **kwargs)
            self.estimator = self.estimator(**varkw)
        if model_url and os.path.isfile(model_url):
            self.model_save_path, mname = os.path.split(model_url)
        elif os.path.isfile(self.model_save_path):
            self.model_save_path, mname = os.path.split(self.model_save_path)
        model_path = FileOps.join_path(self.model_save_path, mname)
        if model_url:
            model_path = FileOps.download(model_url, model_path)
        self.has_load = True
        if not (hasattr(self.estimator, "load")
                and os.path.exists(model_path)):
            return
        return self.estimator.load(model_url=model_path)

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        raise NotImplementedError

    def get_weights(self):
        """Get the weights."""
        raise NotImplementedError

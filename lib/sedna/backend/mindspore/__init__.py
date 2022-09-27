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

import mindspore.context as context
from sedna.backend.base import BackendBase
from sedna.common.file_ops import FileOps


class MSBackend(BackendBase):
    def __init__(self, estimator, fine_tune=True, **kwargs):
        super(MSBackend, self).__init__(estimator=estimator,
                                        fine_tune=fine_tune,
                                        **kwargs)
        self.framework = "mindspore"
        if self.use_npu:
            context.set_context(mode=context.GRAPH_MODE,
                                device_target="Ascend")
        elif self.use_cuda:
            context.set_context(mode=context.GRAPH_MODE,
                                device_target="GPU")
        else:
            context.set_context(mode=context.GRAPH_MODE,
                                device_target="CPU")

        if callable(self.estimator):
            self.estimator = self.estimator()

    def train(self, train_data, valid_data=None, **kwargs):
        if callable(self.estimator):
            self.estimator = self.estimator()
        if self.fine_tune and FileOps.exists(self.model_save_path):
            self.finetune()
        self.has_load = True
        varkw = self.parse_kwargs(self.estimator.train, **kwargs)
        return self.estimator.train(train_data=train_data,
                                    valid_data=valid_data,
                                    **varkw)

    def predict(self, data, **kwargs):
        if not self.has_load:
            self.load()
        varkw = self.parse_kwargs(self.estimator.predict, **kwargs)
        return self.estimator.predict(data=data, **varkw)

    def evaluate(self, data, **kwargs):
        if not self.has_load:
            self.load()
        varkw = self.parse_kwargs(self.estimator.evaluate, **kwargs)
        return self.estimator.evaluate(data, **varkw)

    def finetune(self):
        """todo: no support yet"""

    def load_weights(self):
        model_path = FileOps.join_path(self.model_save_path, self.model_name)
        if os.path.exists(model_path):
            self.estimator.load_weights(model_path)

    def get_weights(self):
        """todo: no support yet"""

    def set_weights(self, weights):
        """todo: no support yet"""

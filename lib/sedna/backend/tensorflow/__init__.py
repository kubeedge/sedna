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

import numpy as np
import tensorflow as tf

from sedna.backend.base import BackendBase
from sedna.common.file_ops import FileOps


if hasattr(tf, "compat"):
    # version 2.0 tf
    ConfigProto = tf.compat.v1.ConfigProto
    Session = tf.compat.v1.Session
    reset_default_graph = tf.compat.v1.reset_default_graph
else:
    # version 1
    ConfigProto = tf.ConfigProto
    Session = tf.Session
    reset_default_graph = tf.reset_default_graph


class TFBackend(BackendBase):
    """Tensorflow Framework Backend base Class"""

    def __init__(self, estimator, fine_tune=True, **kwargs):
        super(TFBackend, self).__init__(
            estimator=estimator, fine_tune=fine_tune, **kwargs)
        self.framework = "tensorflow"

        sess_config = (self._init_gpu_session_config()
                       if self.use_cuda else self._init_cpu_session_config())
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = Session(config=sess_config)
        if callable(self.estimator):
            self.estimator = self.estimator()

    @staticmethod
    def _init_cpu_session_config():
        sess_config = ConfigProto(allow_soft_placement=True)
        return sess_config

    @staticmethod
    def _init_gpu_session_config():
        sess_config = ConfigProto(
            log_device_placement=True, allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7
        sess_config.gpu_options.allow_growth = True
        return sess_config

    def train(self, train_data, valid_data=None, **kwargs):
        if callable(self.estimator):
            self.estimator = self.estimator()
        if self.fine_tune and FileOps.exists(self.model_save_path):
            self.finetune()
        self.has_load = True
        varkw = self.parse_kwargs(self.estimator.train, **kwargs)
        return self.estimator.train(
            train_data=train_data,
            valid_data=valid_data,
            **varkw
        )

    def predict(self, data, **kwargs):
        if not self.has_load:
            reset_default_graph()
            self.load()
        varkw = self.parse_kwargs(self.estimator.predict, **kwargs)
        return self.estimator.predict(data=data, **varkw)

    def evaluate(self, data, **kwargs):
        if not self.has_load:
            reset_default_graph()
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

    def model_info(self, model, relpath=None, result=None):
        ckpt = os.path.dirname(model)
        _, _type = os.path.splitext(model)
        if relpath:
            _url = FileOps.remove_path_prefix(model, relpath)
            ckpt_url = FileOps.remove_path_prefix(ckpt, relpath)
        else:
            _url = model
            ckpt_url = ckpt
        _type = _type.lstrip(".").lower()
        results = [{
                "format": _type,
                "url": _url,
                "metrics": result
            }]
        if _type == "pb":  # report ckpt path when model save as pb file
            results.append({
                "format": "ckpt",
                "url": ckpt_url,
                "metrics": result
            })
        return results


class KerasBackend(TFBackend):
    """Keras Framework Backend base Class"""

    def __init__(self, estimator, fine_tune=True, **kwargs):
        super(TFBackend, self).__init__(
            estimator=estimator, fine_tune=fine_tune, **kwargs)
        self.framework = "keras"

    def set_session(self):
        from keras.backend.tensorflow_backend import set_session
        set_session(self.sess)

    def finetune(self):
        return self.load_weights()

    def get_weights(self):
        return list(map(lambda x: x.tolist(), self.estimator.get_weights()))

    def set_weights(self, weights):
        weights = [np.array(x) for x in weights]
        self.estimator.set_weights(weights)

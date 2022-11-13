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

"""Framework Backend class."""

import os
import warnings

from sedna.common.config import BaseConfig


def set_backend(estimator=None, config=None):
    """Create Trainer class"""
    if estimator is None:
        return
    if config is None:
        config = BaseConfig()
    use_cuda = False
    backend_type = os.getenv(
        'BACKEND_TYPE', config.get("backend_type", "UNKNOWN")
    )
    backend_type = str(backend_type).upper()
    device_category = os.getenv(
        'DEVICE_CATEGORY', config.get("device_category", "CPU")
    )
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'GPU'
        use_cuda = True
    else:
        os.environ['DEVICE_CATEGORY'] = device_category

    if backend_type == "TENSORFLOW":
        from sedna.backend.tensorflow import TFBackend as REGISTER
    elif backend_type == "KERAS":
        from sedna.backend.tensorflow import KerasBackend as REGISTER
    elif backend_type == "TORCH":
        from sedna.backend.torch import TorchBackend as REGISTER
    elif backend_type == "MINDSPORE":
        from sedna.backend.mindspore import MSBackend as REGISTER
    else:
        warnings.warn(f"{backend_type} Not Support yet, use itself")
        from sedna.backend.base import BackendBase as REGISTER

    model_save_url = config.get("model_url")
    base_model_save = config.get("base_model_url") or model_save_url
    model_save_name = config.get("model_name")

    return REGISTER(
        estimator=estimator, use_cuda=use_cuda,
        model_save_path=base_model_save,
        model_name=model_save_name,
        model_save_url=model_save_url
    )

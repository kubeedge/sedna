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
import warnings
from sedna.common.config import BaseConfig
"""Framework Backend class."""


def set_backend(estimator=None, config=None):
    """Create Trainer clss."""
    if estimator is None:
        return
    if config is None:
        config = BaseConfig()
    use_cuda = False
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'GPU'
        use_cuda = True
    elif 'NPU-VISIBLE-DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'NPU'
        os.environ['ORIGIN_RANK_TABLE_FILE'] = os.environ['RANK_TABLE_FILE']
        os.environ['ORIGIN_RANK_SIZE'] = os.environ['RANK_SIZE']

    if config.get("device_category"):
        os.environ['DEVICE_CATEGORY'] = config.get("device_category")
    if config.is_tf_backend():
        from sedna.backend.tensorflow import TFBackend as REGISTER
    elif config.is_kr_backend():
        from sedna.backend.tensorflow import KerasBackend as REGISTER
    else:
        backend_type = config.get("backend_type") or "UNKNOWN"
        warnings.warn(f"{backend_type} Not Support yet, use itself")
        from sedna.backend.base import BackendBase as REGISTER
    model_save_url = config.get("model_url")
    base_model_save = config.get("base_model_save") or model_save_url
    return REGISTER(estimator=estimator, use_cuda=use_cuda,
                    model_save_path=base_model_save,
                    model_save_url=model_save_url)

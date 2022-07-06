
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

from abc import ABC
from enum import Enum
import threading
from sedna.common.log import LOGGER
from abc import ABC, abstractmethod

from enum import Enum
import os

import threading

from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.config import BaseConfig

from sedna.common.file_ops import FileOps
from sedna.common.log import LOGGER

from sedna.core.multi_edge_inference.utils import get_parameters

MODEL_NOT_FOUND = "MODEL_UNKNOWN"


# Class defining the possible plugin services.
class PLUGIN(Enum):
    REID_MANAGER = "ReIDManager"
    REID_MANAGER_I = "ReIDManager_I"
    REID = "ReID_Server"
    REID_I = "ReID_I"
    FEATURE_EXTRACTION = "Feature_Extraction"
    FEATURE_EXTRACTION_I = "Feature_Extraction_I"
    VIDEO_ANALYTICS = "VideoAnalytics"
    VIDEO_ANALYTICS_I = "VideoAnalytics_I"


class PluggableNetworkService(ABC):
    """
    Abstract class to wrap a REST service.
    """
    def __init__(self, ip, port, plugin_api: object = None):
        self.ip = ip
        self.port = port
        self.plugin_api = plugin_api

        assert self.__class__.__name__ in PLUGIN._value2member_map_, \
            f'Plugin {self.__class__.__name__} is non registered!'

        self.kind = PLUGIN(self.__class__.__name__).name

        self._post_init()

        LOGGER.info(
            f"Created PluggableNetworkService of kind {self.kind} \
                with IP {self.ip} and port {self.port}"
            )

    def _post_init(self):
        # If the plugin is hosted, we are starting it as a server exposing
        # an API (in a separate thread). If the plugin is NOT hosted, we
        # already have everything (the interface).
        if self.plugin_api is not None:
            start = getattr(self.plugin_api, "start", None)
            if callable(start):
                threading.Thread(
                    target=self.plugin_api.start, daemon=True
                    ).start()


class PluggableModel(ABC):
    """
    Abstract class to wrap and AI model.
    """
    def __init__(self) -> None:
        self.config = BaseConfig()
        self.model_backend = self._set_backend()

        LOGGER.info(f"Loading model {self.model_name}")
        if not os.path.exists(self.model_path):
            raise FileExistsError(f"Cannot find model: {self.model_path}")
        else:
            self.load()

        LOGGER.info(f"Evaluating model {self.model_name}")
        self.evaluate()

    @property
    def model_path(self):
        if os.path.isfile(self.config.model_url):
            return self.config.model_url
        return get_parameters('model_path') or FileOps.join_path(
            self.config.model_url, self.model_backend.model_name)

    @property
    def model_name(self):
        if os.path.isfile(self.config.model_url):
            return os.path.basename(self.config.model_url)
        else:
            MODEL_NOT_FOUND

    @abstractmethod
    def load(self, **kwargs):
        self.model = self.model_backend.load(**kwargs)

    @abstractmethod
    def update_plugin(self, update_object, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        self.model_backend.evaluate()

    def train(self, **kwargs):
        raise NotImplementedError

    def inference(self, data=None, post_process=None, **kwargs):
        """Calls the model 'predict' function"""
        res = self.model_backend.predict(data, **kwargs)
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        return callback_func(res) if callback_func else res

    def evaluate(self, data, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)
        res = self.model_backend.evaluate(data=data, **kwargs)
        return callback_func(res) if callback_func else res

    def _set_backend(self):
        """Configure AI backend parameters based on model type."""
        use_cuda = False
        backend_type = os.getenv(
            'BACKEND_TYPE', self.config.get("backend_type", "UNKNOWN")
        )
        backend_type = str(backend_type).upper()
        device_category = os.getenv(
            'DEVICE_CATEGORY', self.config.get("device_category", "CPU")
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
        else:
            LOGGER.warn(f"{backend_type} Not Support yet, use itself")
            from sedna.backend.base import BackendBase as REGISTER

        model_save_url = self.config.get("model_url")
        base_model_save = self.config.get("base_model_url") or model_save_url
        model_save_name = self.config.get("model_name")

        return REGISTER(
            estimator=self, use_cuda=use_cuda,
            model_save_path=base_model_save,
            model_name=model_save_name,
            model_save_url=model_save_url
        )

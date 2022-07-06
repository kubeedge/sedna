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

import importlib

from sedna.common.log import LOGGER

def str_to_class(module_name=".", class_name="ByteTracker"):
    """Return a class type from a string reference"""
    LOGGER.info(f"Dynamically loading class {class_name}")
    try:
        module_ = importlib.import_module(module_name + class_name.lower(), package="model")
        try:
            class_ = getattr(module_, class_name)
        except AttributeError:
            LOGGER.error('Class does not exist')
    except ImportError:
        LOGGER.error('Module does not exist')
    return class_ or None
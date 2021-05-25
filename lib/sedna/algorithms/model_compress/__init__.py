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

from sedna.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CPR, alias="defaultCompress")
class XXXX:
    def __init__(self):
        pass

    def compress(self, big_model, data=None, **kwargs) -> dict:

        return {
            "big": big_model, "small": None, "Hem": None
        }

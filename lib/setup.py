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

from setuptools import setup

setup(
    name='sedna',
    version='0.0.1',
    description="The sedna package is designed to help developers \
                better use open source frameworks such as tensorflow \
                on Sedna project",
    packages=['sedna'],
    python_requires='>=3.6',
    install_requires=[
        'flask>=1.1.2',
        'keras>=2.4.3',
        'Pillow>=8.0.1',
        'opencv-python>=4.4.0.44',
        'websockets>=8.1'
        'requests>=2.24.0'
    ]
)

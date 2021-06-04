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

"""Setuptools of sedna"""
from setuptools import setup, find_packages
import sys
import os

assert sys.version_info >= (3, 6), "Sorry, Python < 3.6 is not supported."

with open("README.md", "r") as fh:
    long_desc = fh.read()

with open(os.path.join(os.path.dirname(__file__), 'sedna', 'VERSION'),
          "r", encoding="utf-8") as fh:
    __version__ = fh.read().strip()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [line.strip() for line in
                        fh.readlines() if line.strip()]

setup(
    name='sedna',
    version=__version__,
    description="The sedna package is designed to help developers \
                better use open source frameworks such as tensorflow \
                on Sedna project",
    packages=find_packages(exclude=["tests", "*.tests",
                                    "*.tests.*", "tests.*"]),
    author="",
    author_email="",
    maintainer="",
    maintainer_email="",
    include_package_data=True,
    python_requires=">=3.6",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/kubeedge/sedna",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=install_requires,
    extras_require={
        "tf": ["tensorflow>=1.0.0,<2.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0,<2.0"],
        "pytorch": ["torch==0.4.0", "torchvision==0.2.1"],
        "ms": ["mindspore==1.1.1"],
        "sklearn": ["pandas>=0.25.0", "scikit-learn==0.24.1"]
    },
)

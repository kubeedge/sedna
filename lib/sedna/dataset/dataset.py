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

"""
TODO: the returned Dataset object requires some design:
choice 1: should be compatible with tensorflow.data.Dataset
choice 2: a high level Dataset object not compatible with tensorflow,
but it's unified in our framework.
"""
import logging
import os

from sedna.common.config import BaseConfig

LOG = logging.getLogger(__name__)


def _load_dataset(dataset_url, format, preprocess_fun=None, **kwargs):
    if dataset_url is None:
        LOG.warning(f'dataset_url is None, please check the url.')
        return None
    if format == 'txt':
        LOG.info(
            f"dataset format is txt, now loading txt from [{dataset_url}]")
        samples = _load_txt_dataset(dataset_url)
        if preprocess_fun:
            new_samples = [preprocess_fun(s) for s in samples]
        else:
            new_samples = samples
        return new_samples


def load_train_dataset(data_format, preprocess_fun=None, **kwargs):
    """
    :param data_format: txt
    :param kwargs:
    :return: Dataset
    """
    return _load_dataset(BaseConfig.train_dataset_url, data_format,
                         preprocess_fun, **kwargs)


def load_test_dataset(data_format, preprocess_fun=None, **kwargs):
    """
    :param data_format: txt
    :param kwargs:
    :return: Dataset
    """
    return _load_dataset(BaseConfig.test_dataset_url, data_format,
                         preprocess_fun, **kwargs)


def _load_txt_dataset(dataset_url):
    LOG.info(f'dataset_url is {dataset_url}, now reading dataset_url')

    # use original dataset url,
    # see https://github.com/kubeedge/sedna/issues/35
    root_path = os.path.dirname(BaseConfig.original_dataset_url or dataset_url)
    with open(dataset_url) as f:
        lines = f.readlines()
    new_lines = [root_path + os.path.sep + l for l in lines]
    return new_lines

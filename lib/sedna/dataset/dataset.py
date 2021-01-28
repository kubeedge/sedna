"""
TODO: the returned Dataset object requires some design:
choice 1: should be compatible with tensorflow.data.Dataset
choice 2: a high level Dataset object not compatible with tensorflow,
but it's unified in our framework.
"""

import fileinput
import logging
import os

import numpy as np

from sedna.common.config import BaseConfig

LOG = logging.getLogger(__name__)


def _load_dataset(dataset_url, format, **kwargs):
    if dataset_url is None:
        LOG.warning(f'dataset_url is None, please check the url.')
        return None
    if format == 'txt':
        LOG.info(
            f"dataset format is txt, now loading txt from [{dataset_url}]")
        if kwargs.get('with_image'):
            return _load_txt_dataset_with_image(dataset_url)
        else:
            return _load_txt_dataset(dataset_url)


def load_train_dataset(data_format, **kwargs):
    """
    :param data_format: txt
    :param kwargs:
    :return: Dataset
    """
    return _load_dataset(BaseConfig.train_dataset_url, data_format, **kwargs)


def load_test_dataset(data_format, **kwargs):
    """
    :param data_format: txt
    :param kwargs:
    :return: Dataset
    """
    return _load_dataset(BaseConfig.test_dataset_url, data_format, **kwargs)


def _load_txt_dataset(dataset_url):
    LOG.info(f'dataset_url is {dataset_url}, now reading dataset_url')
    root_path = BaseConfig.data_path_prefix
    with open(dataset_url) as f:
        lines = f.readlines()
    new_lines = [root_path + os.path.sep + l for l in lines]
    return new_lines


def _load_txt_dataset_with_image(dataset_url):
    import keras.preprocessing.image as img_preprocessing
    root_path = os.path.dirname(dataset_url)
    img_data = []
    img_label = []
    for line in fileinput.input(dataset_url):
        file_path, label = line.split(',')
        file_path = (file_path.replace("\\", os.path.sep)
                     .replace("/", os.path.sep))
        file_path = os.path.join(root_path, file_path)
        img = img_preprocessing.load_img(file_path).resize((128, 128))
        img_data.append(img_preprocessing.img_to_array(img) / 255.0)
        img_label += [(0, 1)] if int(label) == 0 else [(1, 0)]
    data_set = [(np.array(line[0]), np.array(line[1]))
                for line in zip(img_data, img_label)]
    return data_set

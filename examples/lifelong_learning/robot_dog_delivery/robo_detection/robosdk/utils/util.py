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
"""This script contains some common tools."""
import os
import sys
import json
import platform
from typing import Dict
from copy import deepcopy
from importlib import import_module
from functools import wraps

import cv2
import yaml
import math
import numpy as np


def singleton(cls):
    """Set class to singleton class.

    :param cls: class
    :return: instance
    """
    __instances__ = {}

    @wraps(cls)
    def get_instance(*args, **kw):
        """Get class instance and save it into glob list."""
        if cls not in __instances__:
            __instances__[cls] = cls(*args, **kw)
        return __instances__[cls]

    return get_instance


def get_machine_type() -> str:
    return str(platform.machine()).lower()


def _url2dict(arg):
    if arg.endswith('.yaml') or arg.endswith('.yml'):
        with open(arg) as f:
            raw_dict = yaml.load(f, Loader=yaml.FullLoader)
    elif arg.endswith('.py'):
        module_name = os.path.basename(arg)[:-3]
        config_dir = os.path.dirname(arg)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        raw_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        sys.modules.pop(module_name)
    elif arg.endswith(".json"):
        with open(arg) as f:
            raw_dict = json.load(f)
    else:
        try:
            raw_dict = json.loads(arg, encoding="utf-8")
        except json.JSONDecodeError:
            raise Exception('config file must be yaml or py')
    return raw_dict


def _dict2config(config, dic):
    """Convert dictionary to config.

    :param Config config: config
    :param dict dic: dictionary

    """
    if isinstance(dic, dict):
        for key, value in dic.items():
            if isinstance(value, dict):
                config[key] = Config()
                _dict2config(config[key], value)
            else:
                config[key] = value


class Config(dict):
    """A Config class is inherit from dict.

    Config class can parse arguments from a config file
    of yaml, json or pyscript.
    :param args: tuple of Config initial arguments
    :type args: tuple of str or dict
    :param kwargs: dict of Config initial argumnets
    :type kwargs: dict
    """

    def __init__(self, *args, **kwargs):
        """Init config class with multiple config files or dictionary."""
        super(Config, self).__init__()
        for arg in args:
            if isinstance(arg, str):
                _dict2config(self, _url2dict(arg))
            elif isinstance(arg, dict):
                _dict2config(self, arg)
            else:
                raise TypeError('args is not dict or str')
        if kwargs:
            _dict2config(self, kwargs)

    def update_obj(self, update: Dict):

        for k, v in update.items():
            orig = getattr(self, k, Config({}))
            if isinstance(orig, dict):
                orig = Config(orig)
            target = deepcopy(v)
            if isinstance(target, (Config, dict)):
                orig.update_obj(target)
                setattr(self, k, orig)
            else:
                setattr(self, k, target)

    def to_json(self, f_out):
        with open(f_out, "w", encoding="utf-8") as fh:
            json.dump(dict(self), fh, indent=4)

    def to_yaml(self, f_out):
        with open(f_out, "w", encoding="utf-8") as fh:
            yaml.dump(dict(self), fh, default_flow_style=False)

    def __call__(self, *args, **kwargs):
        """Call config class to return a new Config object.

        :return: a new Config object.
        :rtype: Config

        """
        return Config(self, *args, **kwargs)

    def __setstate__(self, state):
        """Set state is to restore state from the unpickled state values.

        :param dict state: the `state` type should be the output of
             `__getstate__`.

        """
        _dict2config(self, state)

    def __getstate__(self):
        """Return state values to be pickled.

        :return: change the Config to a dict.
        :rtype: dict

        """
        d = dict()
        for key, value in self.items():
            if isinstance(value, Config):
                value = value.__getstate__()
            d[key] = value
        return d

    def __getattr__(self, key):
        """Get a object attr by its `key`.

        :param str key: the name of object attr.
        :return: attr of object that name is `key`.
        :rtype: attr of object.

        """
        if key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        """Get a object attr `key` with `value`.

        :param str key: the name of object attr.
        :param value: the `value` need to set to target object attr.
        :type value: attr of object.

        """
        self[key] = value

    def __delattr__(self, key):
        """Delete a object attr by its `key`.

        :param str key: the name of object attr.

        """
        del self[key]

    def __deepcopy__(self, memo):
        """After `deepcopy`, return a Config object.

        :param dict memo: same to deepcopy `memo` dict.
        :return: a deep copyed self Config object.
        :rtype: Config object

        """
        return Config(deepcopy(dict(self)))


class ImageQualityEval:

    @classmethod
    def brenner(cls, img):
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 2):
            for y in range(0, shape[1]):
                out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
        return out

    @classmethod
    def laplacian(cls, img):
        return cv2.Laplacian(img, cv2.CV_64F).var()

    @classmethod
    def smd(cls, img):
        shape = np.shape(img)
        out = 0
        for x in range(1, shape[0] - 1):
            for y in range(0, shape[1]):
                out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
                out += math.fabs(int(img[x, y] - int(img[x + 1, y])))
        return out

    @classmethod
    def smd2(cls, img):
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1] - 1):
                out += math.fabs(
                    int(img[x, y]) - int(img[x + 1, y])) * math.fabs(
                    int(img[x, y] - int(img[x, y + 1])))
        return out

    @classmethod
    def variance(cls, img):
        out = 0
        u = np.mean(img)
        shape = np.shape(img)
        for x in range(0, shape[0]):
            for y in range(0, shape[1]):
                out += (img[x, y] - u) ** 2
        return out

    @classmethod
    def energy(cls, img):
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1] - 1):
                out += ((int(img[x + 1, y]) - int(img[x, y])) ** 2) * (
                            (int(img[x, y + 1] - int(img[x, y]))) ** 2)
        return out

    @classmethod
    def vollath(cls, img):
        shape = np.shape(img)
        u = np.mean(img)
        out = -shape[0] * shape[1] * (u ** 2)
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1]):
                out += int(img[x, y]) * int(img[x + 1, y])
        return out

    @classmethod
    def entropy(cls, img):
        out = 0
        count = np.shape(img)[0] * np.shape(img)[1]
        p = np.bincount(np.array(img).flatten())
        for i in range(0, len(p)):
            if p[i] != 0:
                out -= p[i] * math.log(p[i] / count) / count
        return out

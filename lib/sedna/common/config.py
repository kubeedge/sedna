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

# Copy from https://github.com/huawei-noah/vega/blob/master/zeus/common/config.py  # noqa
# We made a re-modify due to vega is exceed out needs

import os
import sys
import yaml
import json
from copy import deepcopy
from importlib import import_module
from inspect import ismethod, isfunction

from .utils import singleton

__all__ = ('Context', 'BaseConfig', )


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


class ConfigSerializable(object):
    """Seriablizable config base class."""

    __original__value__ = None

    @property
    def __allattr__(self):
        attrs = filter(
            lambda attr: not (
                attr.startswith("__") or ismethod(
                    getattr(
                        self,
                        attr)) or isfunction(
                    getattr(
                        self,
                        attr))),
            dir(self))
        return list(attrs)

    def update(self, **kwargs):
        for attr in self.__allattr__:
            if attr not in kwargs:
                continue
            setattr(self, attr, kwargs[attr])

    def to_json(self):
        """Serialize config to a dictionary."""

        attr_dict = {}
        for attr in self.__allattr__:
            value = getattr(self, attr)
            if isinstance(value, type) and isinstance(
                    value(), ConfigSerializable):
                value = value().to_json()
            elif isinstance(value, ConfigSerializable):
                value = value.to_json()
            attr_dict[attr] = value
        return Config(deepcopy(attr_dict))

    def dict(self):
        attr_dict = {}
        for attr in self.__allattr__:
            value = getattr(self, attr)
            if isinstance(value, type) and isinstance(
                    value(), ConfigSerializable):
                value = value().dict()
            elif isinstance(value, ConfigSerializable):
                value = value.dict()
            attr_dict[attr] = value
        return attr_dict

    def __getitem__(self, item):
        return getattr(self, item, None)

    def get(self, item, default=""):
        return self.__getitem__(item) or default

    @classmethod
    def from_json(cls, data):
        """Restore config from a dictionary or a file."""
        if not data:
            return cls
        if cls.__name__ == "ConfigSerializable":
            return cls
        config = Config(deepcopy(data))
        for attr in config:
            if not hasattr(cls, attr):
                setattr(cls, attr, config[attr])
                continue
            class_value = getattr(cls, attr)
            config_value = config[attr]
            if isinstance(class_value, ConfigSerializable) and hasattr(
                    config_value, 'from_json'):
                setattr(cls, attr, class_value.from_json(config_value))
            else:
                setattr(cls, attr, config_value)
        return cls


@singleton
class BaseConfig(ConfigSerializable):
    """The base config"""
    device_category = os.getenv('DEVICE_CATEGORY', 'CPU')  # device category
    # ML framework backend
    backend_type = os.getenv('BACKEND_TYPE', 'TENSORFLOW')
    # local control server
    lc_server = os.getenv("LC_SERVER", "http://127.0.0.1:9100")
    # dataset
    original_dataset_url = os.getenv("ORIGINAL_DATASET_URL")
    train_dataset_url = os.getenv("TRAIN_DATASET_URL")
    test_dataset_url = os.getenv("TEST_DATASET_URL")
    data_path_prefix = os.getenv("DATA_PATH_PREFIX", "/home/data")
    # k8s crd info
    namespace = os.getenv("NAMESPACE", "")
    worker_name = os.getenv("WORKER_NAME", "")
    # the name of JointInferenceService and others Service
    service_name = os.getenv("SERVICE_NAME", "")
    # the name of FederatedLearningJob and others Job
    job_name = os.getenv("JOB_NAME", "sedna")

    pretrained_model_url = os.getenv("PRETRAINED_MODEL_URL", "./")
    model_url = os.getenv("MODEL_URL")
    model_name = os.getenv("MODEL_NAME")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    transmitter = os.getenv("TRANSMITTER", "ws")
    agg_data_path = os.getenv("AGG_DATA_PATH", "./")
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", "")
    access_key_id = os.getenv("ACCESS_KEY_ID", "")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY", "")

    # user parameter
    parameters = os.getenv("PARAMETERS")

    def __init__(self):
        if self.parameters:
            self.parameter = _url2dict(self.parameters)


class Context:
    """The Context provides the capability of obtaining the context"""
    parameters = os.environ

    @classmethod
    def get_parameters(cls, param, default=None):
        """get the value of the key `param` in `PARAMETERS`,
        if not exist, the default value is returned"""
        value = cls.parameters.get(
            param) or cls.parameters.get(str(param).upper())
        return value if value else default

    @classmethod
    def get_algorithm_from_api(cls, algorithm, **param) -> dict:
        """get the algorithm and parameter from api"""
        hard_example_name = cls.get_parameters(f'{algorithm}_NAME')
        hem_parameters = cls.get_parameters(f'{algorithm}_PARAMETERS')
        if not hard_example_name:
            return {}
        try:
            hem_parameters = json.loads(hem_parameters)
            hem_parameters = {
                p["key"]: p.get("value", "")
                for p in hem_parameters if "key" in p
            }
        except Exception:
            hem_parameters = {}

        hem_parameters.update(**param)

        hard_example_mining = {
            "method": hard_example_name,
            "param": hem_parameters
        }

        return hard_example_mining

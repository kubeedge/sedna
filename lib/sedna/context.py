import json
import logging

from sedna.common.config import BaseConfig

LOG = logging.getLogger(__name__)


def parse_parameters(parameters):
    """
    :param parameters:
    [{"key":"batch_size","value":"32"},
    {"key":"learning_rate","value":"0.001"},
    {"key":"min_node_number","value":"3"}]
    ---->
    :return:
    {'batch_size':32, 'learning_rate':0.001, 'min_node_number'=3}
    """
    p = {}
    if parameters is None or len(parameters) == 0:
        LOG.info(f"PARAMETERS={parameters}, return empty dict.")
        return p
    j = json.loads(parameters)
    for d in j:
        p[d.get('key')] = d.get('value')
    return p


class Context:
    """The Context provides the capability of obtaining the context of the
    `PARAMETERS` and `HEM_PARAMETERS` field"""

    def __init__(self):
        self.parameters = parse_parameters(BaseConfig.parameters)
        self.hem_parameters = parse_parameters(BaseConfig.hem_parameters)

    def get_context(self):
        return self.parameters

    def get_parameters(self, param, default=None):
        """get the value of the key `param` in `PARAMETERS`,
        if not exist, the default value is returned"""
        value = self.parameters.get(param)
        return value if value else default

    def get_hem_parameters(self, param, default=None):
        """get the value of the key `param` in `HEM_PARAMETERS`,
        if not exist, the default value is returned"""
        value = self.hem_parameters.get(param)
        return value if value else default


context = Context()

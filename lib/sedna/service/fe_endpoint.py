import pickle
from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult
from sedna.service.client import http_request
from copy import deepcopy

class FE:
    """Endpoint to trigger the Feature Extraction"""

    def __init__(self, service_name, version="",
                 host="127.0.0.1", port="8080", protocol="http"):
        self.server_name = f"{service_name}{version}"
        self.endpoint = f"{protocol}://{host}:{port}/{service_name}"

    def check_server_status(self):
        return http_request(url=self.endpoint, method="GET")

    def feature_extraction(self, x : DetTrackResult, **kwargs):
        """Transfer feature vector to FE worker"""
        # json_data = deepcopy(kwargs)
        # json_data.update({"data": [x.to_json()]})
        _url = f"{self.endpoint}/feature_extraction"
        return http_request(url=_url, method="POST", data=pickle.dumps(x))
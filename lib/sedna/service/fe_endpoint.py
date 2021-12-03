import pickle
from sedna.service.client import http_request


class FE:
    """Endpoint to trigger the Feature Extraction"""

    def __init__(self, service_name, version="",
                 host="127.0.0.1", port="8080", protocol="http"):
        self.server_name = f"{service_name}{version}"
        self.endpoint = f"{protocol}://{host}:{port}/{service_name}"

    def feature_extraction(self, x, **kwargs):
        """Transfer feature vector to FE worker"""
        _url = f"{self.endpoint}/feature_extraction"
        return http_request(url=_url, method="POST", data=pickle.dumps(x))


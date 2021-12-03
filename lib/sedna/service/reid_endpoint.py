import pickle
from sedna.service.client import http_request

class ReID:
    """Endpoint to trigger the ReID"""

    def __init__(self, service_name, version="",
                 host="127.0.0.1", port="8080", protocol="http"):
        self.server_name = f"{service_name}{version}"
        self.endpoint = f"{protocol}://{host}:{port}/{service_name}"

    def check_server_status(self):
        return http_request(url=self.endpoint, method="GET")

    def reid(self, x, **kwargs):
        """Transfer feature vector to ReID worker"""
        _url = f"{self.endpoint}/reid"
        return http_request(url=_url, method="POST", data=pickle.dumps(x))

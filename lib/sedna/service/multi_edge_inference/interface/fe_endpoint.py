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

import pickle

from sedna.service.client import http_request


class FE:
    """Endpoint to trigger the Feature Extraction"""

    def __init__(self, service_name, version="",
                 ip="127.0.0.1", port="8080", protocol="http"):
        self.server_name = f"{service_name}{version}"
        self.endpoint = f"{protocol}://{ip}:{port}/{service_name}"

    def check_server_status(self):
        return http_request(url=self.endpoint, method="GET")

    def transmit(self, data, **kwargs):
        """Transfer feature vector to FE worker"""
        _url = f"{self.endpoint}/feature_extraction"
        return http_request(url=_url, method="POST", data=pickle.dumps(data))

    def get_target_features(self, data, **kwargs):
        """Send target images to FE service and receive back
            the ReID features"""
        _url = f"{self.endpoint}/get_target_features"
        response = http_request(
            url=_url,
            method="POST",
            data=pickle.dumps(data),
            no_decode=True)
        return pickle.loads(response.content)

    def update_service(self, data, **kwargs):
        _url = f"{self.endpoint}/update_service"
        return http_request(url=_url, method="POST", data=pickle.dumps(data))

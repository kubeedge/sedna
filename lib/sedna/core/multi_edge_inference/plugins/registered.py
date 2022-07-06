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

from sedna.common.utils import get_host_ip
from sedna.core.multi_edge_inference.plugins import PluggableNetworkService
from sedna.core.multi_edge_inference.utils import get_parameters
from sedna.service.multi_edge_inference.interface import *
from sedna.service.multi_edge_inference.server import *

"""
Each PluggableNetworkService can be separated into server and interface.
A server runs on a pod/service that IMPLEMENTS the functionalities exposed
by the API. An interface runs on any pods that want to USE the API implemented
by the server.

For example:

POD/SERVICE A ---transmit--->  POD/SERVICE B

In this case, A needs the interface of B. B implements the server that A
wants to access. Bidirectional communication is possible as a service can
implement AND access multiple plugins.

"""


class ReID_Server(PluggableNetworkService):
    def __init__(
        self,
        ip=get_parameters("REID_MODEL_BIND_URL", get_host_ip()),
        port=get_parameters("REID_MODEL_BIND_PORT", "5000"),
        wrapper=None
    ):

        super().__init__(
            ip,
            port,
            ReIDServer(wrapper, "reid", ip, int(port))
        )


class ReID_I(PluggableNetworkService):
    def __init__(
        self,
        ip=get_parameters("REID_MODEL_BIND_URL", "reid-reid"),
        port=get_parameters("REID_MODEL_BIND_PORT", "5000")
    ):

        super().__init__(
            ip,
            port,
            ReID_Endpoint("reid", ip, port=int(port))
        )


class Feature_Extraction(PluggableNetworkService):
    def __init__(
        self,
        ip=get_parameters("FE_MODEL_BIND_URL", get_host_ip()),
        port=get_parameters("FE_MODEL_BIND_PORT", "6000"),
        wrapper=None
    ):

        super().__init__(
            ip,
            port,
            FEServer(wrapper, "feature_extraction", ip, int(port))
        )


class Feature_Extraction_I(PluggableNetworkService):
    def __init__(
        self,
        ip=get_parameters("FE_MODEL_BIND_URL", "feature-extraction-fe"),
        port=get_parameters("FE_MODEL_BIND_PORT", "6000")
    ):

        super().__init__(
            ip,
            port,
            FE("feature_extraction", ip=ip, port=int(port))
        )


class VideoAnalytics(PluggableNetworkService):
    def __init__(
        self,
        ip=get_parameters("DET_MODEL_BIND_URL", get_host_ip()),
        port=get_parameters("DET_MODEL_BIND_PORT", "4000"),
        wrapper=None
    ):

        super().__init__(
            ip,
            port,
            DetectionServer(
                wrapper,
                service_name="video_analytics",
                ip=ip,
                port=int(port)),
        )


class VideoAnalytics_I(PluggableNetworkService):
    def __init__(
        self,
        ip=get_parameters(
            "DET_MODEL_BIND_URL", "video-analytics-videoanalytics"),
        port=get_parameters("DET_MODEL_BIND_PORT", "4000")
    ):

        super().__init__(
            ip,
            port,
            Detection("video_analytics", ip=ip, port=int(port))
        )

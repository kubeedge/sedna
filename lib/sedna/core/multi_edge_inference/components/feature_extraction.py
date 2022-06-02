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

from typing import List
from sedna.core.multi_edge_inference.components import BaseService
from sedna.core.multi_edge_inference.plugins \
    import PLUGIN, PluggableModel, PluggableNetworkService
from sedna.core.multi_edge_inference.plugins.registered \
    import Feature_Extraction, VideoAnalytics_I


class FEService(BaseService):
    """
    In MultiEdgeInference, the Feature Extraction component
    is deployed in the edge or the cloud and it used to extract
    ReID features from frames received by the ObjectDetector component
    and send back to it the enriched data using Kafka or REST API.

    Parameters
    ----------
    consumer_topics : List
        A list of Kafka topics used to communicate with the Object
        Detector service (to receive data from it).
        This is accessed only if the Kafka backend is in use.
    producer_topics : List
        A list of Kafka topics used to communicate with the Object
        Detector service (to send data to it).
        This is accessed only if the Kafka backend is in use.
    plugins : List
        A list of PluggableNetworkService. It can be left empty
        as the FeatureExtraction service is already preconfigured
        to connect to the correct network services.
    models : List
        A list of PluggableModel. By passing a specific instance
        of the model, it is possible to customize the FeatureExtraction
        component to, for example, extract differently the objects
        features.
    timeout: int
        It sets a timeout condition to terminate the main fetch loop
        after the specified amount of seconds has passed since we
        received the last frame.
    asynchronous: bool
        If True, the AI processing will be decoupled from the data
        acquisition step. If False, the processing will be sequential.
        In general, set it to True when ingesting a stream (e.g., RTSP)
        and to False when reading from disk (e.g., a video file).


    Examples
    --------
    model = FeatureExtractionAI() # A class implementing the PluggableModel
    abstract class (example pedestrian_tracking/feature_extraction/worker.py)

    fe_service = FEService(models=[model], asynchronous=False)

    Notes
    -----
    For the parameters described above, only 'models' has to be defined, while
    for others the default value will work in most cases.
    """

    def __init__(
        self,
        consumer_topics=["object_detection"],
        producer_topics=["enriched_object"],
        plugins: List[PluggableNetworkService] = [],
        models: List[PluggableModel] = [],
        timeout=10,
        asynchronous=False
    ):

        merged_plugins =  \
            [VideoAnalytics_I(), Feature_Extraction(wrapper=self)] + plugins

        super().__init__(
            consumer_topics,
            producer_topics,
            merged_plugins,
            models,
            timeout,
            asynchronous)

    def process_data(self, ai, data, **kwargs):
        for ai in self.models:
            result = ai.inference(data)

        if result != []:
            if self.kafka_enabled:
                for d in result:
                    self.producer.write_result(d)
            else:
                plg = self.get_plugin(PLUGIN.VIDEO_ANALYTICS_I)
                plg.plugin_api.transmit(result, **kwargs)

    def update_operational_mode(self, status):
        pass

    def get_target_features(self, ldata):
        # TODO: Fix this workaround, we need a function to select a model
        # based on its name
        fe_ai = self.models[0]
        return fe_ai.get_target_features(ldata)

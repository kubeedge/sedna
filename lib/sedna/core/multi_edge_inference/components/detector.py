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

import time
from typing import List
import numpy as np

from sedna.algorithms.optical_flow import LukasKanade, LukasKanadeCUDA
from sedna.common.log import LOGGER

from sedna.core.multi_edge_inference.plugins \
    import PLUGIN, PluggableModel, PluggableNetworkService
from sedna.core.multi_edge_inference.plugins.registered \
    import Feature_Extraction_I, VideoAnalytics
from sedna.core.multi_edge_inference.components \
    import BaseService, FileOperations


class ObjectDetector(BaseService, FileOperations):
    """
    In MultiEdgeInference, the Object Detection/Tracking component
    is deployed as a service at the edge and it used to detect or
    track objects (for example, pedestrians) and send the result
    to the cloud for further processing using Kafka or REST API.

    Parameters
    ----------
    consumer_topics : List
        A list of Kafka topics used to communicate with the Feature
        Extraction service (to receive data from it).
        This is accessed only if the Kafka backend is in use.
    producer_topics : List
        A list of Kafka topics used to communicate with the Feature
        Extraction service (to send data to it).
        This is accessed only if the Kafka backend is in use.
    plugins : List
        A list of PluggableNetworkService. It can be left empty
        as the ObjectDetector service is already preconfigured
        to connect to the correct network services.
    models : List
        A list of PluggableModel. By passing a specific instance
        of the model, it is possible to customize the ObjectDetector
        to, for example, track different objects as long as the
        PluggableModel interface is respected.
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
    model = ByteTracker() # A class implementing the PluggableModel abstract
    class (example in pedestrian_tracking/detector/model/bytetracker.py)
    objecttracking_service = ObjectDetector(models=[model], asynchronous=True)

    Notes
    -----
    For the parameters described above, only 'models' has to be defined, while
    for others the default value will work in most cases.
    """

    def __init__(
        self,
        consumer_topics=["enriched_object"],
        producer_topics=["object_detection"],
        plugins: List[PluggableNetworkService] = [],
        models: List[PluggableModel] = [],
        timeout=10,
        asynchronous=False
    ):

        merged_plugins = \
            [VideoAnalytics(wrapper=self), Feature_Extraction_I()] + plugins

        super().__init__(
            consumer_topics,
            producer_topics,
            merged_plugins,
            models,
            timeout,
            asynchronous)

        if self.models[0].device == "cuda":
            self.optical_flow = LukasKanadeCUDA()
        else:
            self.optical_flow = LukasKanade()
        self.prev_frame = np.empty(0)

        self.heartbeat = time.time()
        self.data_counter = 0

    def process_data(self, ai, data, **kwargs):
        result = ai.inference(data)

        if result != []:
            if self.kafka_enabled:
                for d in result:
                    self.producer.write_result(d)
            else:
                plg = self.get_plugin(PLUGIN.FEATURE_EXTRACTION_I)
                plg.plugin_api.transmit(result, **kwargs)

        return

    # We change the preprocess function to add the optical flow analysis
    def preprocess(self, data):
        # TODO: Improve this check, this is not reliable.
        if isinstance(data, List):
            self.data_counter += len(data)
            LOGGER.info(
                f"Received data from FE module (counter={self.data_counter}).\
                Writing to local storage"
                )
            self.write_to_disk(data, folder='/data/network_shared/reid/')
            self.heartbeat = time.time()
            return None

        if self.prev_frame.size:
            if self.optical_flow(self.prev_frame, data[0]):
                LOGGER.debug("Movement detected")
                return data
        else:
            self.prev_frame = data[0]
            return data

        return None

    def close(self):
        LOGGER.debug("Perform housekeeping operations.")
        if self.kafka_enabled:
            self.consumer.consumer.close()
            self.producer.producer.close()

    def update_operational_mode(self, status):
        return

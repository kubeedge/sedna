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
from sedna.common.log import LOGGER

from sedna.core.multi_edge_inference.components \
    import BaseService, FileOperations
from sedna.core.multi_edge_inference.plugins \
    import PLUGIN, PluggableModel, PluggableNetworkService
from sedna.core.multi_edge_inference.plugins.registered \
    import Feature_Extraction_I, ReID_Server


class ReID(BaseService, FileOperations):
    """
    In MultiEdgeInference, the ReID component is deployed in the cloud
    and it used to identify a target by compairing its features
    with the ones genereated from the Feature Extraction component.

    Parameters
    ----------
    consumer_topics : List
        Leave empty.
    producer_topics : List
        Leave empty.
    plugins : List
        A list of PluggableNetworkService. It can be left empty
        as the ReID component is already preconfigured
        to connect to the correct network services.
    models : List
        A list of PluggableModel. In this case we abuse of the term
        model as the ReID doesn't really use an AI model but rather
        a wrapper for the ReID functions.
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
    model = ReIDWorker() # A class implementing the PluggableModel abstract
    class (example in pedestrian_tracking/reid/worker.py)

    self.job = ReID(models=[model], asynchronous=False)

    Notes
    -----
    For the parameters described above, only 'models' has to be defined, while
    for others the default value will work in most cases.
    """

    def __init__(
        self,
        consumer_topics=[],
        producer_topics=[],
        plugins: List[PluggableNetworkService] = [],
        models: List[PluggableModel] = [],
        timeout=10,
        asynchronous=True
    ):

        self.models = models
        merged_plugins =  \
            [ReID_Server(wrapper=self), Feature_Extraction_I()] + plugins

        super().__init__(
            consumer_topics,
            producer_topics,
            merged_plugins,
            models,
            timeout,
            asynchronous)

    def _post_init(self):
        super()._post_init()
        self.update_operational_mode(None)

    def process_data(self, ai, data, **kwargs):
        result = ai.predict(data)

        # if result != []:
        #     self.write_to_disk(result, folder='/data/processed/')

    def update_operational_mode(self, status):
        for ai in self.models:
            try:
                ldata = ai.update_plugin(status)
                target_list = self.get_target_features(ldata)
                ai.update_target(target_list)
            except Exception as ex:
                LOGGER.error(
                    f"Unable to update AI parameters/configuration for \
                        service {ai.__class__.__name__}. [{ex}]"
                    )
        return

    def get_target_features(self, ldata):
        feature_extraction_plugin = \
            self.get_plugin(PLUGIN.FEATURE_EXTRACTION_I)
        features = \
            feature_extraction_plugin.plugin_api.get_target_features(ldata)
        return features

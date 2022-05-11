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
from sedna.core.multi_edge_tracking.components import BaseService
from sedna.core.multi_edge_tracking.plugins import PLUGIN, PluggableModel, PluggableNetworkService
from sedna.core.multi_edge_tracking.plugins.registered import Feature_Extraction, VideoAnalytics_I

class FEService(BaseService):
    """
   Feature Extraction service.
   """

    def __init__(self, consumer_topics = ["object_detection"], producer_topics=["enriched_object"], plugins: List[PluggableNetworkService] = [], models: List[PluggableModel] = [], timeout = 10, asynchronous = False):
        merged_plugins =  [VideoAnalytics_I(), Feature_Extraction(wrapper=self)] + plugins
        super().__init__(consumer_topics, producer_topics, merged_plugins, models, timeout, asynchronous)

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
        # TODO: Fix this workaround, we need a function to select a model based on its name
        fe_ai = self.models[0]
        return fe_ai.get_target_features(ldata)
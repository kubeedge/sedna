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

from interface import fedavg, s3_transmitter, simple_chooser
from interface import Estimator
from sedna.service.server import AggregationServerV2
from sedna.common.config import BaseConfig

def run_server():
    estimator = Estimator()
    estimator.saved = BaseConfig.model_url

    server = AggregationServerV2(
        data=None, # mistnet, train, test
        estimator=estimator,
        aggregation=fedavg,
        transmitter=s3_transmitter,
        chooser=simple_chooser)

    server.start()


if __name__ == '__main__':
    run_server()

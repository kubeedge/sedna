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

from sedna.common.config import Context
from sedna.service.server import AggregationServer


def run_server():
    aggregation_algorithm = Context.get_parameters(
        "aggregation_algorithm", "FedAvg"
    )
    exit_round = int(Context.get_parameters(
        "exit_round", 3
    ))
    agg_ip = Context.get_parameters("AGG_IP", "0.0.0.0")
    agg_port = int(Context.get_parameters("AGG_PORT", "7363"))
    server = AggregationServer(
        servername=aggregation_algorithm,
        host=agg_ip,
        http_port=agg_port,
        exit_round=exit_round
    )
    server.start()


if __name__ == '__main__':
    run_server()

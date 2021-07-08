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

from sedna.core.federated_learning import FLWorker
from torch import nn
import asyncio
from interface import Trainer

if __name__ == '__main__':
    # main()
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    trainer = Trainer(model=model)
    client = FLWorker(model=model, trainer=trainer)
    client.configure()
    # for 3.7
    # asyncio.run(client.start_client())
    # for 3.6
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start_client())
    

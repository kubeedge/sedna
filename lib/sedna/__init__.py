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

import logging

from . import joint_inference, federated_learning, incremental_learning
from .context import context
from .dataset.dataset import load_train_dataset, load_test_dataset


def log_configure():
    logging.basicConfig(
        format='[%(asctime)s][%(name)s][%(levelname)s][%(lineno)s]: '
               '%(message)s',
        level=logging.INFO)


LOG = logging.getLogger(__name__)

log_configure()

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

import numpy as np

import sedna.ml_model
from sedna.ml_model import load_model

LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    valid_data = sedna.load_test_dataset(data_format="txt", with_image=True)

    x_valid = np.array([tup[0] for tup in valid_data])
    y_valid = np.array([tup[1] for tup in valid_data])

    loaded_model = load_model()
    LOG.info(f"x_valid is {loaded_model.predict(x_valid)}")

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

import os

import numpy as np
import keras.preprocessing.image as img_preprocessing

from sedna.common.config import Context
from sedna.datasources import TxtDataParse
from sedna.core.federated_learning import FederatedLearning

from interface import Estimator


def image_process(line):
    file_path, label = line.split(',')
    original_dataset_url = Context.get_parameters('original_dataset_url')
    root_path = os.path.dirname(original_dataset_url)
    file_path = os.path.join(root_path, file_path)
    img = img_preprocessing.load_img(file_path).resize((128, 128))
    data = img_preprocessing.img_to_array(img) / 255.0
    label = [0, 1] if int(label) == 0 else [1, 0]
    data = np.array(data)
    label = np.array(label)
    return [data, label]


def main():
    fl_instance = FederatedLearning(estimator=Estimator)

    # load dataset.

    test_data = TxtDataParse(data_type="test", func=image_process)
    test_data.parse(fl_instance.config.test_dataset_url)

    return fl_instance.inference(test_data.x)


if __name__ == '__main__':
    print(main())

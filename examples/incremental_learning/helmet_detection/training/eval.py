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

from sedna.common.config import Context
from sedna.core.incremental_learning import IncrementalLearning
from sedna.datasources import TxtDataParse
from interface import Estimator

max_epochs = 1


def _load_txt_dataset(dataset_url):

    # use original dataset url,
    # see https://github.com/kubeedge/sedna/issues/35
    original_dataset_url = Context.get_parameters('original_dataset_url')
    return original_dataset_url + os.path.sep + dataset_url


def main():
    # load dataset.
    test_dataset_url = Context.get_parameters('test_dataset_url')

    valid_data = TxtDataParse(data_type="test", func=_load_txt_dataset)
    valid_data.parse(test_dataset_url, use_raw=True)

    # read parameters from deployment config.
    class_names = Context.get_parameters("class_names")
    class_names = [label.strip() for label in class_names.split(',')]
    input_shape = Context.get_parameters("input_shape")
    input_shape = tuple(int(shape) for shape in input_shape.split(','))

    model = IncrementalLearning(estimator=Estimator)
    return model.evaluate(valid_data, class_names=class_names,
                          input_shape=input_shape)


if __name__ == '__main__':
    main()

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

import sedna
from validate_utils import validate

LOG = logging.getLogger(__name__)
max_epochs = 1


def main():
    # load dataset.
    test_data = sedna.load_test_dataset(data_format='txt', with_image=False)

    # read parameters from deployment config.
    class_names = sedna.context.get_parameters("class_names")
    class_names = [label.strip() for label in class_names.split(',')]
    input_shape = sedna.context.get_parameters("input_shape")
    input_shape = tuple(int(shape) for shape in input_shape.split(','))

    model = validate

    sedna.incremental_learning.evaluate(model=model,
                                          test_data=test_data,
                                          class_names=class_names,
                                          input_shape=input_shape)


if __name__ == '__main__':
    main()

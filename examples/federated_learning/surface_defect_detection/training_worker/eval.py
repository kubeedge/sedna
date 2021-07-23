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

from sedna.datasources import TxtDataParse
from sedna.core.federated_learning import FederatedLearning

from interface import Estimator, image_process


def main():
    fl_instance = FederatedLearning(estimator=Estimator)

    # load dataset.

    valid_data = TxtDataParse(data_type="valid", func=image_process)
    valid_data.parse(fl_instance.config.test_dataset_url)

    return fl_instance.evaluate(valid_data)


if __name__ == '__main__':
    print(main())

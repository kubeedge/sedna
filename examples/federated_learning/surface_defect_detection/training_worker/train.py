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
from sedna.common.config import Context, BaseConfig
from sedna.core.federated_learning import FederatedLearning

from interface import Estimator, image_process


def main():
    # load dataset.
    train_dataset_url = BaseConfig.train_dataset_url
    test_dataset_url = BaseConfig.test_dataset_url

    train_data = TxtDataParse(data_type="train", func=image_process)
    train_data.parse(train_dataset_url)

    valid_data = TxtDataParse(data_type="test", func=image_process)
    valid_data.parse(test_dataset_url)

    epochs = int(Context.get_parameters("epochs", 1))
    batch_size = int(Context.get_parameters("batch_size", 1))
    aggregation_algorithm = Context.get_parameters(
        "aggregation_algorithm", "FedAvg"
    )
    learning_rate = float(
        Context.get_parameters("learning_rate", 0.001)
    )
    validation_split = float(
        Context.get_parameters("validation_split", 0.2)
    )

    fl_model = FederatedLearning(
        estimator=Estimator,
        aggregation=aggregation_algorithm)

    train_jobs = fl_model.train(
        train_data=train_data,
        valid_data=valid_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split
    )
    return train_jobs


if __name__ == '__main__':
    main()

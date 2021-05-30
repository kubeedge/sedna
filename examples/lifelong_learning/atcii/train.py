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

from interface import DATACONF, Estimator, feature_process
from sedna.common.config import Context, BaseConfig
from sedna.datasources import CSVDataParse

from sedna.core.lifelong_learning import LifelongLearning


def main():
    # load dataset.
    train_dataset_url = BaseConfig.train_dataset_url
    train_data = CSVDataParse(data_type="train", func=feature_process)
    train_data.parse(train_dataset_url, label=DATACONF["LABEL"])

    early_stopping_rounds = int(Context.get_parameters("early_stopping_rounds", 100))
    metric_name = Context.get_parameters("metric_name", "mlogloss")
    ll_job = LifelongLearning(
        estimator=Estimator,
        task_definition="TaskDefinitionByDataAttr",
        task_definition_param='{"attribute": ["Season", "Cooling startegy_building level"]}'
    )
    train_experiment = ll_job.train(
        train_data=train_data,
        metric_name=metric_name,
        early_stopping_rounds=early_stopping_rounds
    )

    return train_experiment


if __name__ == '__main__':
    print(main())

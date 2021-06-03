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

import json

from sedna.datasources import CSVDataParse
from sedna.common.config import Context, BaseConfig
from sedna.core.lifelong_learning import LifelongLearning

from interface import DATACONF, Estimator, feature_process


def main():
    test_dataset_url = BaseConfig.test_dataset_url
    valid_data = CSVDataParse(data_type="valid", func=feature_process)
    valid_data.parse(test_dataset_url, label=DATACONF["LABEL"])
    attribute = json.dumps({"attribute": DATACONF["ATTRIBUTES"]})
    model_threshold = float(Context.get_parameters('model_threshold', 0))

    ll_job = LifelongLearning(
        estimator=Estimator,
        task_definition="TaskDefinitionByDataAttr",
        task_definition_param=attribute
    )
    eval_experiment = ll_job.evaluate(
        data=valid_data, metrics="precision_score",
        metrics_param={"average": "micro"},
        model_threshold=model_threshold
    )
    return eval_experiment


if __name__ == '__main__':
    print(main())

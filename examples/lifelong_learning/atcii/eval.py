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
from sedna.common.config import Context
from sedna.datasources import CSVDataParse

from sedna.core.lifelong_learning import LifelongLearning


def main():
    method_selection = {
        "task_definition": "TaskDefinitionByDataAttr",
        "task_definition_param": '{"attribute": ["Season"]}',

    }
    test_dataset_url = Context.get_parameters('test_dataset_url')
    valid_data = CSVDataParse(data_type="test", func=feature_process)
    valid_data.parse(test_dataset_url, label=DATACONF["LABEL"])
    ll_model = LifelongLearning(estimator=Estimator,
                                method_selection=method_selection)
    return ll_model.evaluate(data=valid_data,
                             metrics="precision_score",
                             metics_param={"average": "micro"})


if __name__ == '__main__':
    print(main())

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

import csv
import time
from interface import DATACONF, Estimator, feature_process
from sedna.common.config import Context
from sedna.datasources import CSVDataParse
from sedna.core.lifelong_learning import LifelongLearning


def main():
    method_selection = {
        "task_definition": "TaskDefinitionByDataAttr",
        "task_definition_param": '{"attribute": ["Season"]}',

    }

    ll_model = LifelongLearning(estimator=Estimator,
                                method_selection=method_selection)

    infer_dataset_url = Context.get_parameters('infer_dataset_url')
    file_handle = open(infer_dataset_url, "r", encoding="utf-8")
    header = list(csv.reader([file_handle.readline().strip()]))[0]
    infer_data = CSVDataParse(data_type="test", func=feature_process)

    while 1:
        where = file_handle.tell()
        line = file_handle.readline()
        if not line:
            time.sleep(1)
            file_handle.seek(where)
            continue
        reader = list(csv.reader([line.strip()]))

        rows = dict(zip(header, reader[0]))
        infer_data.parse(rows, label=DATACONF["LABEL"])

        print(ll_model.inference(infer_data))


if __name__ == '__main__':
    print(main())

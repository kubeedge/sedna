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
import csv
import json
import time

from sedna.common.config import Context
from sedna.datasources import CSVDataParse
from sedna.core.lifelong_learning import LifelongLearning

from interface import DATACONF, Estimator, feature_process


def main():

    utd = Context.get_parameters("UTD_NAME", "TaskAttr")
    attribute = json.dumps({"attribute": DATACONF["ATTRIBUTES"]})
    utd_parameters = Context.get_parameters("UTD_PARAMETERS", {})
    ut_saved_url = Context.get_parameters("UTD_SAVED_URL", "/tmp")

    task_mining = {
        "method": "TaskMiningByDataAttr",
        "param": attribute
    }

    unseen_task_detect = {
        "method": utd,
        "param": utd_parameters
    }

    ll_job = LifelongLearning(
        estimator=Estimator,
        task_mining=task_mining,
        unseen_task_detect=unseen_task_detect)

    infer_dataset_url = Context.get_parameters('infer_dataset_url')
    file_handle = open(infer_dataset_url, "r", encoding="utf-8")
    header = list(csv.reader([file_handle.readline().strip()]))[0]
    infer_data = CSVDataParse(data_type="test", func=feature_process)

    unseen_sample = open(os.path.join(ut_saved_url, "unseen_sample.csv"),
                         "w", encoding="utf-8")
    unseen_sample.write("\t".join(header + ['pred']) + "\n")
    output_sample = open(f"{infer_dataset_url}_out.csv", "w", encoding="utf-8")
    output_sample.write("\t".join(header + ['pred']) + "\n")

    while 1:
        where = file_handle.tell()
        line = file_handle.readline()
        if not line:
            time.sleep(1)
            file_handle.seek(where)
            continue
        reader = list(csv.reader([line.strip()]))
        rows = reader[0]
        data = dict(zip(header, rows))
        infer_data.parse(data, label=DATACONF["LABEL"])
        rsl, is_unseen, target_task = ll_job.inference(infer_data)

        rows.append(list(rsl)[0])

        output = "\t".join(map(str, rows)) + "\n"
        if is_unseen:
            unseen_sample.write(output)
        output_sample.write(output)
    unseen_sample.close()
    output_sample.close()


if __name__ == '__main__':
    print(main())

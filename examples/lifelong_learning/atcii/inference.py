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

    utd = Context.get_parameters("UTD_NAME", "TaskAttr")
    utd_parameters = Context.get_parameters("UTD_PARAMETERS", {})
    ut_saved_url = Context.get_parameters("UTD_SAVED_URL", "/tmp")

    ll_job = LifelongLearning(
        estimator=Estimator,
        task_mining="TaskMiningByDataAttr",
        task_mining_param='{"attribute": ["Season", "Cooling startegy_building level"]}',
        unseen_task_detect=utd,
        unseen_task_detect_param=utd_parameters
    )

    infer_dataset_url = Context.get_parameters('infer_dataset_url')
    file_handle = open(infer_dataset_url, "r", encoding="utf-8")
    header = list(csv.reader([file_handle.readline().strip()]))[0]
    infer_data = CSVDataParse(data_type="test", func=feature_process)

    unseen_sample = open(infer_dataset_url, "w", encoding="utf-8")
    unseen_sample.write("\t".join(header + ['pred', 'TaskAttr', 'SampleAttr']) + "\n")
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

        task_attr = target_task.samples.meta_attr
        sample_attr = target_task.model.meta_attr

        rows.extend([list(rsl)[0], task_attr, sample_attr])
        if is_unseen:
            unseen_sample.write("\t".join(map(str, rows)) + "\n")
    unseen_sample.close()


if __name__ == '__main__':
    print(main())

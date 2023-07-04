# Copyright 2023 The KubeEdge Authors.
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

from sedna.core.lifelong_learning import LifelongLearning
from sedna.common.config import Context, BaseConfig
from sedna.datasources import TxtDataParse

from interface import Estimator


def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url', "")
    dataset_urls = dataset_url.split()
    dataset_urls = [
        os.path.join(
            os.path.dirname(original_dataset_url),
            dataset_url) for dataset_url in dataset_urls]
    return dataset_urls[:-1], dataset_urls[-1]


def train(estimator, train_data):
    task_definition = {
        "method": "TaskDefinitionByOrigin",
        "param": {
            "attribute": Context.get_parameters("attribute"),
            "city": Context.get_parameters("city")
        }
    }

    task_allocation = {
        "method": "TaskAllocationByOrigin"
    }

    ll_job = LifelongLearning(estimator,
                              task_definition=task_definition,
                              task_relationship_discovery=None,
                              task_allocation=task_allocation,
                              task_remodeling=None,
                              inference_integrate=None,
                              task_update_decision=None,
                              unseen_task_allocation=None,
                              unseen_sample_recognition=None,
                              unseen_sample_re_recognition=None
                              )

    ll_job.train(train_data)


def run():
    estimator = Estimator(num_class=int(Context.get_parameters("num_class", 24)),
                          epochs=int(Context.get_parameters("epoches", 1)))
    train_dataset_url = BaseConfig.train_dataset_url
    train_data = TxtDataParse(data_type="train", func=_load_txt_dataset)
    train_data.parse(train_dataset_url, use_raw=False)

    train(estimator, train_data)


if __name__ == '__main__':
    run()

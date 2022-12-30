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
import time

from PIL import Image
from sedna.datasources import BaseDataSource
from sedna.common.config import Context
from sedna.common.log import LOGGER
from sedna.common.file_ops import FileOps
from sedna.core.lifelong_learning import LifelongLearning

from interface import Estimator


def unseen_sample_postprocess(sample, save_url):
    if isinstance(sample, dict):
        img = sample.get("image")
        image_name = "{}.png".format(str(time.time()))
        image_url = FileOps.join_path(save_url, image_name)
        img.save(image_url)
    else:
        image_name = os.path.basename(sample[0])
        image_url = FileOps.join_path(save_url, image_name)
        FileOps.upload(sample[0], image_url, clean=False)


def preprocess(samples):
    data = BaseDataSource(data_type="test")
    data.x = [samples]
    return data


def init_ll_job():
    estimator = Estimator(num_class=Context.get_parameters("num_class", 24),
                          save_predicted_image=True,
                          merge=True)

    task_allocation = {
        "method": "TaskAllocationStream"
    }
    unseen_task_allocation = {
        "method": "UnseenTaskAllocationDefault"
    }

    ll_job = LifelongLearning(
        estimator,
        unseen_estimator=unseen_task_processing,
        task_definition=None,
        task_relationship_discovery=None,
        task_allocation=task_allocation,
        task_remodeling=None,
        inference_integrate=None,
        task_update_decision=None,
        unseen_task_allocation=unseen_task_allocation,
        unseen_sample_recognition=None,
        unseen_sample_re_recognition=None)
    return ll_job


def unseen_task_processing():
    return "Warning: unseen sample detected."


def predict():
    ll_job = init_ll_job()
    test_data_dir = Context.get_parameters("test_data")
    test_data = os.listdir(test_data_dir)
    test_data_num = len(test_data)
    count = 0

    # Simulate a permenant inference service
    while True:
        for i, data in enumerate(test_data):
            LOGGER.info(f"Start to inference image {i + count + 1}")

            test_data_url = os.path.join(test_data_dir, data)
            img_rgb = Image.open(test_data_url).convert("RGB")
            sample = {'image': img_rgb, "depth": img_rgb, "label": img_rgb}
            predict_data = preprocess(sample)
            prediction, is_unseen, _ = ll_job.inference(
                predict_data, 
                unseen_sample_postprocess=unseen_sample_postprocess)
            LOGGER.info(f"Image {i + count + 1} is unseen task: {is_unseen}")
            LOGGER.info(
                f"Image {i + count + 1} prediction result: {prediction}")
            time.sleep(1.0)

        count += test_data_num


if __name__ == '__main__':
    predict()

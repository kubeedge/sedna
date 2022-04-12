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
from interface import mistnet, s3_transmitter
from interface import Dataset, Estimator
from sedna.common.config import BaseConfig
from sedna.core.federated_learning import FederatedLearningV2
from examples.ms_nnrt.ms_nnrt_models.ms_acl_inference import Inference
from examples.ms_nnrt.ms_nnrt_trainer_yolo import Trainer
from examples.ms_nnrt.ms_nnrt_algorithms.ms_mistnet import Algorithm

def main():
    data = Dataset()
    estimator = Estimator()
    data.parameters["data_path"] = BaseConfig.train_dataset_url.replace("robot.txt", "")
    data.parameters["train_path"] = os.path.join(data.parameters["data_path"], "./coco128/train2017/")
    data.parameters["test_path"] = data.parameters["train_path"]
    data.parameters["train_annFile"] = os.path.join(data.parameters["data_path"], "coco128/annotations/instances_train2017.json")
    if "s3_endpoint_url" in s3_transmitter.parameters:
        from plato.utils import s3
        s3_client = s3.S3(s3_transmitter.parameters["s3_endpoint_url"], s3_transmitter.parameters["access_key"],
                      s3_transmitter.parameters["secret_key"], s3_transmitter.parameters["s3_bucket"])
        #s3_client.download_from_s3("model/client_model/yolov5x_cutlayer4.om", "./yolov5x_cutlayer4.om")
        s3_client.download_from_s3("model/client_model/network_f.om", "./network_f.om")

    estimator.model = Inference(0, "./network_f.om", 320, 320)
    estimator.trainer = Trainer(model=estimator.model)
    estimator.algorithm = Algorithm(estimator.trainer)

    fl_model = FederatedLearningV2(
        data=data,
        estimator=estimator,
        aggregation=mistnet,
        transmitter=s3_transmitter)

    fl_model.train()

if __name__ == '__main__':
    main()

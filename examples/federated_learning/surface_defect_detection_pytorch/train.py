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
from torchvision import transforms
from PIL import Image

from interface import fedavg, s3_transmitter
from interface import myDataset, Estimator
from sedna.core.federated_learning import FederatedLearningV2
from sedna.datasources import TxtDataParse
from sedna.common.config import BaseConfig

def image_process(line):
    file_path, label = line.split(',')
    root_path = os.getcwd()
    file_path = os.path.join(root_path, file_path)
    transform = transforms.Compose([transforms.Resize((128, 128)), 
                                    transforms.PILToTensor()])
    x = Image.open(file_path)
    x = transform(x)/255.
    y = int(label) 

    return [x, y]

def readFromTxt(path):
    data_x = []
    data_y = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            x, y = image_process(line)
            data_x.append(x)
            data_y.append(y)
    return data_x, data_y

def main():
    train_dataset_url = BaseConfig.train_dataset_url
    # we have same data in the trainset and testset
    test_dataset_url = BaseConfig.train_dataset_url

    # train_data = TxtDataParse(data_type="train", func=image_process)
    # train_data.parse(train_dataset_url)
    train_data = readFromTxt(train_dataset_url)
    data = myDataset(trainset=train_data, testset=train_data)
    
    estimator = Estimator()

    fl_model = FederatedLearningV2(
        data=data,
        estimator=estimator,
        aggregation=fedavg,
        transmitter=s3_transmitter)

    fl_model.train()

if __name__ == '__main__':
    main()

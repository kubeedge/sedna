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
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision

from sedna.algorithms.aggregation import FedAvgV2
from sedna.algorithms.client_choose import SimpleClientChoose
from sedna.common.config import Context
from sedna.core.federated_learning import FederatedLearningV2

# os.environ['BACKEND_TYPE'] = 'KERAS'

simple_chooser = SimpleClientChoose(per_round=2)

# It has been determined that mistnet is required here.
fedavg = FedAvgV2()

# The function `get_transmitter_from_config()` returns an object instance.
s3_transmitter = FederatedLearningV2.get_transmitter_from_config()

class SddDataset():
    def __init__(self, x, y):
        self.labels = y
        self.images = x
        self.index = 0
    
    def __len__(self):
        return len(self.images)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index = self.index % len(self.images)
        x, y = self.images[self.index], self.labels[self.index]
        self.index = self.index + 1
        return x, y

    
class myDataset:
    def __init__(self, trainset=None, testset=None):
        self.customized = True
        transform = [
            c_vision.Resize((128, 128)),
            py_vision.ToTensor(),
            py_vision.Normalize((0.5, ), (0.5, ))
        ]
        self.trainset = GeneratorDataset(SddDataset(trainset[0], trainset[1]),
                                         column_names=["image", "label"])
        self.testset = GeneratorDataset(SddDataset(trainset[0], trainset[1]),
                                         column_names=["image", "label"])
        
        self.trainset = self.trainset.map(operations=transform, 
                                          input_columns="image").batch(
                                              batch_size=int(Context.get_parameters("batch_size", 32)))
                                          
        self.testset = self.testset.map(operations=transform, 
                                          input_columns="image").batch(
                                              batch_size=int(Context.get_parameters("batch_size", 32)))

class SddModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Dense(8192, 64)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Dense(64, 32)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Dense(32, 2)

    def construct(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout(self.flatten(x))
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

class Estimator:
    def __init__(self):
        self.model = SddModel()
        self.pretrained = None
        self.saved = None
        self.hyperparameters = {
            "use_mindspore": True,
            "type": "basic",
            "rounds": int(Context.get_parameters("exit_round", 5)),
            "target_accuracy": 0.97,
            "epochs": int(Context.get_parameters("epochs", 5)),
            "batch_size": int(Context.get_parameters("batch_size", 32)),
            "optimizer": "SGD",
            "learning_rate": float(Context.get_parameters("learning_rate", 0.01)),
            # The machine learning model
            "model_name": "sdd_model",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "history": 0.1
        }



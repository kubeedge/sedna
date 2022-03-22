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
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

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

class SddDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.images = x
        self.labels = y
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
class myDataset:
    def __init__(self, trainset=None, testset=None) -> None:
        self.customized = True
        self.trainset = SddDataset(trainset[0], trainset[1])
        self.testset = SddDataset(testset[0], testset[1])

class Estimator:
    def __init__(self) -> None:
        self.model = self.build()
        self.pretrained = None
        self.saved = None
        self.hyperparameters = {
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

    @staticmethod
    def build():
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(6272, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)) 
        return model



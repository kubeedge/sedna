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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Sequential

from sedna.algorithms.aggregation import FedAvgV2
from sedna.algorithms.client_choose import SimpleClientChoose
from sedna.common.config import Context
from sedna.core.federated_learning import FederatedLearningV2

os.environ['BACKEND_TYPE'] = 'KERAS'

simple_chooser = SimpleClientChoose(per_round=2)

# It has been determined that mistnet is required here.
fedavg = FedAvgV2()

# The function `get_transmitter_from_config()` returns an object instance.
s3_transmitter = FederatedLearningV2.get_transmitter_from_config()


class Dataset:
    def __init__(self, trainset=None, testset=None) -> None:
        self.customized = True
        self.trainset = tf.data.Dataset.from_tensor_slices((trainset.x, trainset.y))
        self.trainset = self.trainset.batch(int(Context.get_parameters("batch_size", 32)))
        self.testset = tf.data.Dataset.from_tensor_slices((testset.x, testset.y))
        self.testset = self.testset.batch(int(Context.get_parameters("batch_size", 32)))

class Estimator:
    def __init__(self) -> None:
        self.model = self.build()
        self.pretrained = None
        self.saved = None
        self.hyperparameters = {
            "use_tensorflow": True,
            "is_compiled": True,
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
            "weight_decay": 0.0
        }

    @staticmethod
    def build():
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3),
                         activation="relu", strides=(2, 2),
                         input_shape=(128, 128, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation="softmax"))
        
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        return model

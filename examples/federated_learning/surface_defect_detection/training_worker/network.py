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

from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.models import Sequential


class GlobalModelInspectionCNN:
    def __init__(self):
        pass

    def build_model(self):
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

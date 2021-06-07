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
import keras
import numpy as np
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.models import Sequential

os.environ['BACKEND_TYPE'] = 'KERAS'


class Estimator:
    def __init__(self, **kwargs):
        """Model init"""

        self.model = self.build()
        self.has_init = False

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

    def train(self,
              train_data, valid_data=None,
              epochs=1,
              batch_size=1,
              learning_rate=0.01,
              validation_split=0.2):
        """ Model train """

        if not self.has_init:
            loss = keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = [keras.metrics.categorical_accuracy]
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            self.has_init = True
        x, y = train_data.x, train_data.y
        if valid_data:
            x1, y1 = valid_data.x, valid_data.y
            history = self.model.fit(x, y, epochs=int(epochs),
                                     batch_size=int(batch_size),
                                     validation_data=(x1, y1))
        else:
            history = self.model.fit(x, y, epochs=int(epochs),
                                     batch_size=int(batch_size),
                                     validation_split=validation_split)
        return {k: list(map(np.float, v)) for k, v in history.history.items()}

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def load_weights(self, model):
        if not os.path.isfile(model):
            return
        return self.model.load_weights(model)

    def predict(self, datas):
        return self.model.predict(datas)

    def evaluate(self, test_data, **kwargs):
        pass

    def load(self, model_url):
        return self.load_weights(model_url)

    def save(self, model_path=None):
        """
        save model as a single pb file from checkpoint
        """
        return self.model.save(model_path)

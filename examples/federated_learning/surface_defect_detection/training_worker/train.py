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
from tensorflow import keras

import sedna
from network import GlobalModelInspectionCNN
from sedna.ml_model import save_model


def image_process(line):
    import keras.preprocessing.image as img_preprocessing
    file_path, label = line.split(',')
    img = img_preprocessing.load_img(file_path).resize((128, 128))
    data = img_preprocessing.img_to_array(img) / 255.0
    label = [0, 1] if int(label) == 0 else [1, 0]
    data = np.array(data)
    label = np.array(label)
    return [data, label]


def main():
    # load dataset.
    train_data = sedna.load_train_dataset(data_format="txt",
                                          preprocess_fun=image_process)

    x = np.array([tup[0] for tup in train_data])
    y = np.array([tup[1] for tup in train_data])

    # read parameters from deployment config.
    epochs = sedna.context.get_parameters("epochs")
    batch_size = sedna.context.get_parameters("batch_size")
    aggregation_algorithm = sedna.context.get_parameters(
        "aggregation_algorithm"
    )
    learning_rate = float(
        sedna.context.get_parameters("learning_rate", 0.001)
    )

    model = GlobalModelInspectionCNN().build_model()

    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.categorical_accuracy]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model = sedna.federated_learning.train(
        model=model,
        x=x, y=y,
        epochs=epochs,
        batch_size=batch_size,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        aggregation_algorithm=aggregation_algorithm
    )

    # Save the model based on the config.
    save_model(model)


if __name__ == '__main__':
    main()

import numpy as np
from tensorflow import keras

import neptune
from neptune.ml_model import save_model
from network import GlobalModelInspectionCNN


def main():
    # load dataset.
    train_data = neptune.load_train_dataset(data_format="txt")

    x = np.array([tup[0] for tup in train_data])
    y = np.array([tup[1] for tup in train_data])

    # read parameters from deployment config.
    epochs = neptune.context.get_parameters("epochs")
    batch_size = neptune.context.get_parameters("batch_size")
    aggregation_algorithm = neptune.context.get_parameters(
        "aggregation_algorithm"
    )
    learning_rate = float(
        neptune.context.get_parameters("learning_rate", 0.001)
    )

    model = GlobalModelInspectionCNN().build_model()

    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.categorical_accuracy]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model = neptune.federated_learning.train(
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

import logging

import numpy as np

import neptune.ml_model
from neptune.ml_model import load_model

LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    valid_data = neptune.load_test_dataset(data_format="txt", with_image=True)

    x_valid = np.array([tup[0] for tup in valid_data])
    y_valid = np.array([tup[1] for tup in valid_data])

    loaded_model = load_model()
    LOG.info(f"x_valid is {loaded_model.predict(x_valid)}")

import logging

import numpy as np

from neptune.ml_model import load_model
import neptune.ml_model

LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    valid_data = neptune.load_test_dataset(data_format="txt")

    x_valid = np.array([tup[0] for tup in valid_data])
    y_valid = np.array([tup[1] for tup in valid_data])

    loaded_model = load_model()
    LOG.info(f"x_valid is {loaded_model.predict(x_valid)}")

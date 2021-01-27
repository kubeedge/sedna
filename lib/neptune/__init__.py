import logging

from . import joint_inference, federated_learning, incremental_learning
from .context import context
from .dataset.dataset import load_train_dataset, load_test_dataset


def log_configure():
    logging.basicConfig(
        format='[%(asctime)s][%(name)s][%(levelname)s][%(lineno)s]: '
               '%(message)s',
        level=logging.INFO)


LOG = logging.getLogger(__name__)

log_configure()

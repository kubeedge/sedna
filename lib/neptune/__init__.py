import logging

from . import joint_inference
from .context import context


def log_configure():
    logging.basicConfig(
        format='[%(asctime)s][%(name)s][%(levelname)s][%(lineno)s]: '
               '%(message)s',
        level=logging.INFO)


LOG = logging.getLogger(__name__)

log_configure()

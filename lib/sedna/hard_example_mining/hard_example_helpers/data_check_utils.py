import logging

logger = logging.getLogger(__name__)


def data_check(data):
    """Check the data in [0,1]."""
    return 0 <= float(data) <= 1

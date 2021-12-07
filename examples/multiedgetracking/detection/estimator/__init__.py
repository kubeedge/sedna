import importlib

from sedna.common.log import LOGGER

def str_to_estimator_class(module_name=".", estimator_class="Yolov5"):
    """Return a class type from a string reference"""
    LOGGER.info(f"Dynamically loading estimator class {estimator_class}")
    try:
        module_ = importlib.import_module(module_name + estimator_class.lower(), package="estimator")
        try:
            class_ = getattr(module_, estimator_class)
        except AttributeError:
            LOGGER.error('Estimator class does not exist')
    except ImportError:
        LOGGER.error('Module does not exist')
    return class_ or None
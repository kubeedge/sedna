"""
Integrate the inference results of all related tasks
"""

from typing import List

import numpy as np

from ..artifact import Task


class BaseInferenceIntegrate:
    """
    Base class for default calculation algorithm for inference integration

    Parameters
    ----------
    models: All models used for sample inference
    """

    def __init__(self, models: list, **kwargs):
        self.models = models

    def __call__(self, tasks: List[Task]):
        """
        Parameters
        ----------
        tasks: All tasks with sample result

        Returns
        -------
        result: inference results for all the inference samples
        """
        raise NotImplementedError

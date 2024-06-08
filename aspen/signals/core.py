"""
Sets out the core Signal interface
"""

import abc

import pandas as pd


class ISignal(metaclass=abc.ABCMeta):
    """
    Core Signal Interface
    """

    @abc.abstractmethod
    def calculate(self, **data: pd.DataFrame):
        pass

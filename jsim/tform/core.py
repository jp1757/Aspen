"""
Basic interface for defining transformation logic
"""

import abc

import pandas as pd


class ITForm(metaclass=abc.ABCMeta):
    """
    Interface setting out structure of all transformation objects
    """

    @abc.abstractmethod
    def apply(self, data: pd.Series) -> pd.Series:
        """
        Place transformation logic here

        :param data: (pandas.Series) input data to apply transformation to
        :return: (pandas.Series) transformed data
        """

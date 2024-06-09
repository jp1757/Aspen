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
    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Place transformation logic here

        :param data: (pandas.DataFrame) input data to apply transformation to
        :param *other: (pandas.DataFrame) other data frames to use in transformation
        :return: (pandas.DataFrame) transformed data
        """
        pass

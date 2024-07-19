"""
Backtest interface
"""

import abc

import pandas as pd


class IBTest(metaclass=abc.ABCMeta):
    """
    Sets out the structure that backtest objects should follow
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique backtest id"""
        pass

    @abc.abstractmethod
    def run(self) -> pd.DataFrame:
        """
        Run a backtest

        :return (pd.DataFrame) of asset weights
        """
        pass

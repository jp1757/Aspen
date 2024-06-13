"""
Backtest interface
"""

import abc

import pandas as pd


class IBTest(abc.ABCMeta):
    """
    Sets out the structure that backtest objects should follow
    """

    @abc.abstractmethod
    def run(self) -> pd.DataFrame:
        """
        Run a backtest

        :return (pd.DataFrame) of asset weights
        """
        pass

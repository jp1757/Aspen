"""
Sets out an interface for handling portfolio backtest re-balancing
"""

import abc

import pandas as pd


class IRebal(metaclass=abc.ABCMeta):

    def rebalance(
        self,
        *,
        date: pd.Timestamp,
        signals: pd.DataFrame,
        asset: pd.DataFrame,
    ) -> bool:
        pass

    def finalize(self, weights: pd.DataFrame) -> pd.DataFrame:
        return weights


class AllDates(IRebal):

    def rebalance(
        self,
        *,
        date: pd.Timestamp,
        signals: pd.DataFrame,
        asset: pd.DataFrame,
    ) -> bool:
        return True

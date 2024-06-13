"""
Generic backtest object
"""
from typing import List
import pandas as pd

from aspen.backtest.core import IBTest
from aspen.signals.core import ISignal


class BTest(IBTest):
    """
    A basic backtest object that takes signals, assets & returns
    a set of portfolio holdings
    """

    def __init__(
            self,
            *,
            tr: pd.DataFrame,
            signals: List[ISignal],
            sdate: pd.Timestamp = pd.Timestamp(year=2000, month=1, day=1),
            edate: pd.Timestamp = pd.Timestamp.now().date(),
    ):
        pass

    def run(self) -> pd.DataFrame:
        pass


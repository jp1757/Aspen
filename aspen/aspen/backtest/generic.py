"""
Generic backtest object
"""
from typing import List
import pandas as pd

from aspen.backtest.core import IBTest
from aspen.signals.core import ISignals
from aspen.pcr.core import IPortConstruct

import aspen.tform.lib.asset


class BTest(IBTest):
    """
    A basic backtest object that takes signals, assets & returns
    a set of portfolio holdings
    """

    def __init__(
            self,
            *,
            dates: pd.DatetimeIndex,
            tr: pd.DataFrame,
            signals: ISignals,
            pcr: IPortConstruct,
            normalise: bool = True,
    ):
        # Store instance vars
        self.dates = dates
        self.signals = signals
        self.pcr = pcr
        self.normalise = normalise

        # Get asset returns indexed to input dates
        self.returns = aspen.tform.lib.asset.Returns(dates).apply(tr)
        self.returns_shift = self.returns.shift(-1)

    def run(self) -> pd.DataFrame:
        """
        Run backtest looping through input dates
        :return: (pd.DataFrame) of asset weights through time
        """

        # Calculate signal data
        signals = self.signals.combine(self.normalise)

        weights = [
            self.pcr.weights(
                date=d, signals=signals.loc[:d], asset=self.returns.loc[:d]
            )
            for d in self.dates
        ]

        return weights

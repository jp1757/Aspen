"""
Generic backtest object
"""
from typing import List
import pandas as pd

from aspen.backtest.core import IBTest
from aspen.signals.core import ISignals
from aspen.pcr.core import IPortConstruct

from aspen.tform.library.align import Reindex


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

        # Align total return data to input dates
        self.tr = Reindex(dates).apply(tr)

    def run(self) -> pd.DataFrame:
        """
        Run backtest looping through input dates
        :return: (pd.DataFrame) of asset weights through time
        """

        # Calculate signal data
        signals = self.signals.combine(self.normalise)

        weights = [
            self.pcr.weights(
                date=d, signals=signals.loc[:d], asset=self.tr.loc[:d]
            )
            for d in self.dates
            if len(signals.loc[:d]) > 0
        ]

        wgt_df = pd.concat(weights, axis=1).T
        wgt_df.index.freq = pd.infer_freq(wgt_df.index)

        return wgt_df

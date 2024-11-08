"""
Generic backtest object
"""

import pandas as pd
import functools

from aspen.backtest.core import IBTest
from aspen.signals.core import ISignals, INormalise
from aspen.pcr.core import IPortConstruct
from aspen.pcr.rebalance import IRebal, AllDates

from aspen.library.tform.align import Reindex


class BTest(IBTest):
    """
    A basic backtest object that takes signals, assets & returns
    a set of portfolio holdings
    """

    def __init__(
        self,
        name: str,
        *,
        dates: pd.DatetimeIndex,
        tr: pd.DataFrame,
        signals: ISignals,
        pcr: IPortConstruct,
        normalise: INormalise = None,
        rebalance: IRebal = None,
        signal: str = None,
    ) -> None:
        # Store instance vars
        self._name = name
        self.dates = dates
        self.signals = signals
        self.pcr = pcr
        self.rebalance = rebalance
        self.normalise = normalise
        self.signal = signal

        # Align total return data to input dates
        self.tr = Reindex(dates).apply(tr)

    @property
    def name(self) -> str:
        """Unique backtest id"""
        return self._name

    @functools.lru_cache(maxsize=None)
    def run(self) -> pd.DataFrame:
        """
        Run backtest looping through input dates
        :return: (pd.DataFrame) of asset weights through time
        """

        # Calculate signal data
        signals = self.signals.build(name=self.signal)

        # Normalise
        if self.normalise is not None:
            signals = self.normalise.norm(signals)

        # Set rebalance object
        rebalance = AllDates() if self.rebalance is None else self.rebalance

        weights = [
            self.pcr.weights(date=d, signals=signals.loc[:d], asset=self.tr.loc[:d])
            for d in self.dates
            if (len(signals.loc[:d]) > 0)
            and rebalance.rebalance(
                date=d, signals=signals.loc[:d], asset=self.tr.loc[:d]
            )
        ]

        wgt_df = pd.concat(weights, axis=1).T.dropna(how="all")
        wgt_df.index = pd.to_datetime(wgt_df.index)
        wgt_df.index.freq = pd.infer_freq(wgt_df.index)
        wgt_df.index.name = signals.index.name
        wgt_df.name = self.name

        # Shift weights forward 1-step
        # date  signal  weight
        # t     1       0
        # t+1   1       1
        wgt_df = wgt_df.shift(1)

        # Final chance to rebalance weights
        wgt_df = rebalance.finalize(wgt_df)

        return wgt_df

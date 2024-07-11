"""
Signal interface implementations
"""

import pandas as pd

from aspen.signals.core import ISignal, ISignals


class SMean(ISignals):
    """
    Combine multiple signal objects by taking a cross-sectional mean
    across assets on each date
    """

    def __init__(self, *signals: ISignal):

        self._signals = {s.name: s for s in signals}
        if len(signals) > len(self._signals):
            raise ValueError(
                f"Duplicate signal name suspected: {[s.name for s in signals]}"
            )

    def build(self, name: str = None) -> pd.DataFrame:
        """
        Serve up signal data by either combining multiple signals or
        returning a specific signal by setting the 'name' parameter

        :param name: (str, optional) name of signal to return
        :return: pd.DataFrame of signal data indexed by date with columns set
            to asset ids
        """

        if name is None:
            df = pd.concat([s.calculate() for k, s in self._signals.items()])
            df = df.stack().groupby(level=[0, 1]).mean().unstack().sort_index()

            assert not df.index.duplicated().any(), "Duplicate index values found"

            return df
        else:
            return self._signals[name].calculate()

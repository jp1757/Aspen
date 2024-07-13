"""
Signal interface implementations
"""

import pandas as pd

from aspen.signals.core import ISignal
from aspen.signals.generic import Signals


class SMean(Signals):
    """
    Combine multiple signal objects by taking a cross-sectional mean
    across assets on each date
    """

    def _combine(self) -> pd.DataFrame:
        """Combination logic"""
        df = pd.concat([s.calculate() for k, s in self._signals.items()])
        df = df.stack().groupby(level=[0, 1]).mean().unstack().sort_index()

        assert not df.index.duplicated().any(), "Duplicate index values found"

        return df

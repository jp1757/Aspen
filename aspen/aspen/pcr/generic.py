"""
Portfolio construction implementation that just passes back latest signal values
"""

import pandas as pd

from aspen.pcr.core import IPortConstruct


class PCRPass(IPortConstruct):

    def weights(
        self,
        *,
        date: pd.Timestamp,
        signals: pd.DataFrame,
        asset: pd.DataFrame,
    ) -> pd.Series:
        return signals.iloc[-1]

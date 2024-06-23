"""
Defines a portfolio object
"""
import numpy as np
import pandas as pd

from aspen.tform.library.align import Reindex


class Portfolio(object):
    """
    Portfolio object that takes a set of weights
    & converts to a set of historical portfolio returns
    """

    def __init__(self, tr: pd.DataFrame, weights: pd.DataFrame) -> None:
        # Check assets appear in total returns dataframe
        asset_diff = set(weights.columns) - set(tr.columns)
        if len(asset_diff) > 0:
            raise ValueError(f"No returns data for: [{asset_diff}]")

        self.weights = weights.copy()
        self.asset_tr = tr[self.weights.columns].copy()

        # Calculate returns
        self.returns, self.tr = self._returns()

    def _returns(self) -> pd.DataFrame:
        # Re-index total returns to align with weights
        rtf = Reindex(self.weights.index).apply(self.asset_tr)

        # Calculate pct change & shift to align with weights for correct period
        returns = rtf.pct_change().shift(-1)

        # Multiply by weights & sum for portfolio returns
        port = (returns * self.weights).sum(axis=1).shift()

        # Shift returns forward to re-align with correct period
        port = port.shift(1)

        # Calculate portfolio total return index
        port.iloc[0] = 0
        tr = (1 + port).cumprod()

        port.iloc[0] = np.NaN

        return port, tr

    def drift(self, daily_tr: pd.DataFrame) -> pd.DataFrame:
        pass

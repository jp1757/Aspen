"""
Defines a portfolio object
"""
from typing import Tuple

import numpy as np
import pandas as pd

from aspen.library.tform.align import Align


def returns(
        *, dates: pd.DatetimeIndex, weights: pd.DataFrame, asset_tr: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    # Re-index total return prices & weights to align with dates
    weights = Align(dates, fillforward=True).apply(weights)
    asset_tr = Align(dates, fillforward=True).apply(asset_tr)

    # Calculate returns & shift to align with weights for correct period
    _returns = asset_tr.pct_change().shift(-1)

    # Multiply by weights & sum for portfolio returns
    port = _returns * weights

    # Shift returns forward to re-align with correct period
    port = port.shift(1)

    # Sum returns across assets for portfolio return ensuring at least one non-nan value
    port = port.sum(axis=1, min_count=1)

    # Drop leading successive NaNs leaving one
    start_key = port.isna().cumsum().diff().idxmin()
    start_idx = port.index.get_loc(start_key)
    port = port.iloc[start_idx - 1:].copy()

    # Calculate portfolio total return index
    port.iloc[0] = 0
    tr = (1 + port).cumprod()

    port.iloc[0] = np.NaN

    return port, tr


class Portfolio(object):
    """
    Portfolio object that takes a set of weights
    & converts to a set of historical portfolio returns
    """

    def __init__(
            self, name: str, *, asset_tr: pd.DataFrame, weights: pd.DataFrame
    ) -> None:
        """
        Init portfolio object, calculate returns & the total return index

        :param name: (str) portfolio name
        :param asset_tr: (pd.DataFrame) asset total return prices i.e [1.0, 1.01, 0.98...]
            Column names should be set to assets, dates as index. Assets should match
            those found in weights dataframe
        :param weights: (pd.DataFrame) asset weights to calculate returns over. Index
            set to dates & columns set to equivelent asset names in asset_tr dataframe
        """

        self._name = name

        # Check assets appear in total returns dataframe
        asset_diff = set(weights.columns) - set(asset_tr.columns)
        if len(asset_diff) > 0:
            raise ValueError(f"No returns data for: [{asset_diff}]")

        self.__wgts = weights.copy()
        self.__wgts.name = name
        self.asset_tr = asset_tr[self.weights.columns].copy()

        # Calculate returns
        self.__ret, self.__tr = returns(
            dates=weights.index, weights=weights, asset_tr=asset_tr
        )
        self.__ret.name = name
        self.__tr.name = name

    @property
    def name(self):
        return self._name

    @property
    def weights(self) -> pd.DataFrame:
        """
        Return a dataframe of asset weights through time. Indexed by dates
        with assets as columns
        """
        return self.__wgts

    @property
    def returns(self) -> pd.Series:
        """Returns a series of portfolio returns indexed by date"""
        return self.__ret

    @property
    def tr(self) -> pd.Series:
        """Returns a series of portfolio total return prices indexed by date"""
        return self.__tr

    def drift(self, asset_tr: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drifted asset portfolio weights

        wi,t+1 = wi,t x ((1+Ri,t) / (1+Rp,t))

        (w_{i,t+1}) represents the drifted weight for asset (i) in the next period.
        (R_{i,t}) is the return of asset (i) in the current period.
        (R_{p,t}) is the portfolio return in the current period.

        :param asset_tr: (pd.DataFrame) asset total return price data set to a higher
            frequency than the asset weights passed through __init__. Index set to
            dates, columns set to assets.
            i.e. if the asset weights are monthly then pass something like weekly or
            daily total return prices for this to work.

        :return: (pd.DataFrame) return a dataframe of drifted weights indexed to the
            same date index passed via the asset_tr param.  Index set to dates, columns
            set to assets.
        """

        # Re-index to all days
        dates = pd.date_range(start=asset_tr.index.min(), end=asset_tr.index.max())

        # Calculate returns indexed to higher frequency asset returns filling forward
        # the static weight from the previous period
        _ret, _tr = returns(dates=dates, weights=self.weights, asset_tr=asset_tr)

        # Align weights with asset total return prices
        weights = Align(dates, fillforward=True).apply(self.weights)

        # Shift weights to align with returns
        wgt_shift = weights.shift(1)

        # Calculate asset returns
        asset_ret = asset_tr.pct_change()

        # Calculate inter-period drifted weights
        drift = wgt_shift * ((1 + asset_ret).div(1 + _ret, axis=0))

        # Merge in actual re-balance weights to drift weights
        # 1st remove actual weights dates from drift dataframe
        drift = drift.loc[list(set(drift.index) - set(self.weights.index))]
        # Next concat + merge drift weights with actual weights
        drift = pd.concat([drift, self.weights])
        # Drop rows with all NaNs and sort
        drift = drift.dropna(how="all").sort_index()
        # Set index name
        drift.index.name = self.weights.index.name
        drift.name = self.name

        return drift

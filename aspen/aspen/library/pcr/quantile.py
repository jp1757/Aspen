"""
Portfolio construction implementations for quintile bins
"""
import pandas as pd
import numpy as np

from aspen.pcr.core import IPortConstruct


class QuantileEW(IPortConstruct):
    """
    An object that calculates an equally weighted portfolio with a total
    net weight of zero.  Only assets in the top and bottom bins get weights
    with them being positive and negative respectively
    """

    def __init__(self, *, long_bin: int, short_bin: int, gross: float = 200.0) -> None:
        """
        Init object
        :param long_bin: (int) integer label of bin to allocate positive equal weights to
        :param short_bin: (int) integer label of bin to allocate negative equal weights to
        :param gross: (float, optional) total gross exposure of portfolio weights to
            target, in percent
        """
        self.long_bin = long_bin
        self.short_bin = short_bin
        self.gross = gross / 100
        self.exp = gross / 200

    def weights(
            self, *, date: pd.Timestamp, signals: pd.DataFrame, asset: pd.DataFrame,
    ) -> pd.Series:
        """
        Get weights for latest date

        :param date: (pd.Timestamp) current date for weights
        :param signals: (pd.DataFrame) signal data, indexed by date, assets as columns
        :param asset: (pd.DataFrame) asset data needed for weight calculation.  Returns
            or total returns etc.  Indexed by date, assets as columns

        :return: pd.Series of weights with assets set as index, name set to date
        """

        # Get latest signal data
        sig = signals.iloc[-1]

        # Check date matches
        assert date == sig.name

        # Calculate weights
        wgt_long = self.exp / len(sig[sig == self.long_bin])
        wgt_short = self.exp / len(sig[sig == self.short_bin]) * -1

        weights = pd.Series(
            np.where(
                sig == self.long_bin, wgt_long, np.where(
                    sig == self.short_bin, wgt_short, 0
                )
            ),
            index=sig.index,
            name=sig.name
        )

        np.testing.assert_almost_equal(weights.abs().sum(), self.gross)
        np.testing.assert_almost_equal(
            weights.loc[sig[sig == self.long_bin].index].sum(), self.exp
        )
        np.testing.assert_almost_equal(
            weights.loc[sig[sig == self.short_bin].index].sum(), -self.exp
        )

        return weights

"""
Signals object implementations
"""

import pandas as pd

from aspen.signals.generic import Signals


class XSMean(Signals):
    """
    Calculate the cross-sectional mean value of all signals on each date
    Does not work on multi-indexed column dataframes
    """

    def __init__(self, *args, dropna: str = "any", **kwargs):
        """
        Init XSMean object
        :param args: *args to pass to base class Signals
        :param dropna: (str) drop dates if there are 'any' NaN values or 'all' NaNs
        :param kwargs: **kwargs to pass to base class Signals
        """
        self.dropna = dropna
        super().__init__(*args, **kwargs)

    def _combine(self) -> pd.DataFrame:
        """Calculate mean across all signals on each date"""

        # Check columns not multi-indexed
        for x in self.signals:
            if isinstance(x.calculate(), pd.MultiIndex):
                raise ValueError(
                    f"Signal: {x.name} - columns are of type pd.MultiIndex"
                )

        # Calculate each signal & combine into one dataframe incorporating a multi-index
        # made up of date & asset id, with one column for each signal value
        df = (
            pd.concat(
                [pd.Series(x.calculate().stack(), name=x.name) for x in self.signals],
                axis=1,
            )
            .unstack()
            .ffill()
            .dropna(how=self.dropna)
            .stack()
        )

        # Take a cross-sectional mean across each asset for each date
        sig_mean = df.mean(axis=1).unstack()

        return sig_mean


class XSMeanSimple(Signals):
    """
    Calculate the cross-sectional mean of signals using simple 'sum()' on dataframes
    Any rows containing np.NaN values will have a mean of np.NaN.  Use XSMean if you
    wanted pd.DataFrame().mean(axis=1) functionality - does not work on multi-indexed
    column dataframes
    """

    def _combine(self) -> pd.DataFrame:
        _sum = sum([x.calculate() for x in self.signals])
        return _sum / len(self.signals)

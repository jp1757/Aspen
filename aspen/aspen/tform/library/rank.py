"""
Objects for ranking data
"""

import pandas as pd

from aspen.tform.core import ITForm


class RankXSect(ITForm):
    """
    Calculate cross-sectional ranks on input data with
    the highest values getting the lowest ranks (by default)
    """

    def __init__(self, *, pct: bool = True, ascending: bool = False, **kwargs):
        """
        Init object
        :param pct: (bool, optional) use percentile ranks or ordered numerical
            ranks [1,2,3...n]
        :param ascending: (bool, optional) when False, highest values get the
            lowest rank, when True the lowest values will get the lowest ranks.
            When using percentile ranks a rank value of 1.0 is considered a high rank
        :param kwargs: (optional) any other keyword args to pass to pandas.rank function
        """
        self.pct = pct
        self.ascending = ascending
        self.kwargs = kwargs

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Rank data

        :param data: (pandas.DataFrame) input data to apply transformation to
        :param *other: (pandas.DataFrame) other data frames to use in transformation
        :return: (pandas.DataFrame) transformed data
        """
        return data.rank(axis=1, pct=self.pct, ascending=self.ascending, **self.kwargs)


class QCutXSect(ITForm):
    """
    Split data cross-sectionally into bins using the pandas.qcut function
    """

    def __init__(self, bins: int, **kwargs):
        self.bins = bins
        self.kwargs = kwargs

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Split data into bins

        :param data: (pandas.DataFrame) input data to apply transformation to
        :param *other: (pandas.DataFrame) other data frames to use in transformation
        :return: (pandas.DataFrame) transformed data
        """
        qcuts = [
            pd.qcut(row, q=self.bins, labels=range(1, self.bins + 1), **self.kwargs)
            for idx, row in data.iterrows()
        ]
        qcuts = pd.concat(qcuts, axis=1).T
        qcuts.index.name = data.index.name

        return qcuts

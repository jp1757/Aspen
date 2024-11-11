"""
Objects for ranking data
"""

import pandas as pd

from aspen.tform.core import ITForm
import aspen.library.tform.rank


class RankXSect(ITForm):
    """
    Calculate cross-sectional ranks on input data with
    the highest values getting the lowest ranks (by default)
    """

    def __init__(self, *, pct: bool = True, ascending: bool = False, **kwargs) -> None:
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

    def __init__(self, bins: int, **kwargs) -> None:
        """
        Init qcut TForm
        :param bins: (int) number of bins to map values into
        :param kwargs: (optional) key word args to pass to pandas.qcut function
        """
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


class Quantile(ITForm):
    """
    Use the rank and qcut TForm objects to transform signal data
    into binned data
    """

    def __init__(self, *, rank: ITForm, bins: int, **kwargs) -> None:
        """
        Init Quantile object
        :param rank: (ITForm) a tform object that creates cross-sectional ranked data
        :param bins: (int) number of bins
        :param kwargs: (optional) kwargs to pass to aspen.tform.library.rank.QCutXSect
        """
        self.rank = rank
        self.qcut = aspen.library.tform.rank.QCutXSect(bins=bins, **kwargs)

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise signal data
        :param data: (pandas.DataFrame) input data to apply transformation to
        :param *other: (pandas.DataFrame) other data frames to use in transformation
        :return: (pandas.DataFrame) transformed data
        """
        ranks = self.rank.apply(data)
        return self.qcut.apply(ranks)

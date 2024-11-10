"""
INormalise implementations
"""

import pandas as pd

import aspen.library.tform.rank
from aspen.signals.core import INormalise
from aspen.tform.core import ITForm


class Quantile(INormalise):
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

    def norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise signal data
        :param data: (pd.DataFrame) signal data to normalise
        :return: (pd.DataFrame) normalised signal data
        """
        ranks = self.rank.apply(data)
        return self.qcut.apply(ranks)


class XSWeights(INormalise):
    """Weight proportionally to an input score so absolute values sum to 1.0"""

    def norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise signal data
        :param data: (pd.DataFrame) signal data to normalise
        :return: (pd.DataFrame) normalised signal data
        """
        return data.div(data.abs().sum(axis=1), axis=0)

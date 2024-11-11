"""Scoring transformations"""

import pandas as pd

from aspen.tform.core import ITForm


class ZScore(ITForm):
    """Timeseries zscore calculation"""

    def __init__(self, tform: ITForm) -> None:
        """
        Init object
        :param tform: (ITForm) transformation object for calulating rolling window.
            Typically implements pd.rolling or pd.ewm
        """
        self.tf = tform

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate zscore on each column through time

        :param data: (pandas.DataFrame) input data to apply transformation to.
            Index set to dates and with each column representing a score
        :param *other: (pandas.DataFrame) other data frames to use in transformation
        :return: (pandas.DataFrame) transformed data
        """

        rolling = self.tf.apply(data)

        mean = rolling.mean()
        sd = rolling.std()
        zsc = data.sub(mean).div(sd)

        return zsc


class ZScoreXS(ITForm):
    """Cross-sectional zscore calculation"""

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate zscore cross-sectionally on each date

        :param data: (pandas.DataFrame) input data to apply transformation to.
            Index set to dates and with each column representing a score
        :param *other: (pandas.DataFrame) other data frames to use in transformation
        :return: (pandas.DataFrame) transformed data
        """

        mean = data.mean(axis=1)
        sd = data.std(axis=1)
        zsc = data.sub(mean, axis=0).divide(sd, axis=0)

        return zsc


class XSWeights(ITForm):
    """Weight proportionally to an input score so absolute values sum to 1.0"""

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weights cross-sectionally on each date

        :param data: (pandas.DataFrame) input data to apply transformation to.
            Index set to dates and with each column representing a score
        :param *other: (pandas.DataFrame) other data frames to use in transformation
        :return: (pandas.DataFrame) transformed data
        """
        return data.div(data.abs().sum(axis=1), axis=0).sort_index()

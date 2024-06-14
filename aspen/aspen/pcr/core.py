"""Interface for portfolio construction objects"""

import abc
import pandas as pd


class IPortConstruct(metaclass=abc.ABCMeta):
    """
    Core portfolio construction interface.
    Get weights for current point-in-time.
    """

    @abc.abstractmethod
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
        pass

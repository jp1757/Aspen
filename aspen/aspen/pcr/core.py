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
            self, signals: pd.DataFrame, asset: pd.DataFrame, last: pd.Series = None,
    ) -> pd.Series:
        """
        Get weights for latest date

        :param signals: (pd.DataFrame) signal data, indexed by date, assets as columns
        :param asset: (pd.DataFrame) asset data needed for weight calculation.  Returns
            or total returns etc.  Indexed by date, assets as columns
        :param last: (pd.Series) previous weights, indexed by assets, name set to date

        :return: pd.Series of weights with assets set as index, name set to date
        """
        pass

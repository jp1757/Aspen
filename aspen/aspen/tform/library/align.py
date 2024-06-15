"""
Transformations aligning data sets
"""
import pandas as pd

from aspen.tform.core import ITForm


class Reindex(ITForm):
    """
    Re-index dataframe to a set of specific dates by
    first moving to daily dates then filling forward to
    ensure no data loss
    """

    def __init__(self, dates: pd.DatetimeIndex):
        self.dates = dates

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Place transformation logic here

        :param data: (pandas.DataFrame) input data to re-index
        :param *other: (pandas.DataFrame) no used
        :return: (pandas.DataFrame) transformed data
        """

        # Create all days between min & max of input data
        all_dates = pd.date_range(data.index.min(), data.index.max(), freq="D")

        # Reindex to all days
        data = data.reindex(all_dates, method="ffill")

        # Reindex to dates passed to __init__
        return data.reindex(self.dates, method="ffill")


class Align(ITForm):
    """
    Align multiple dataframes
    """

    def __init__(self, *dates: pd.DatetimeIndex, fillforward: bool = True) -> None:
        """
        Init object to align
        :param dates: (pd.DatetimeIndex) dates of input dataframes to align
        """

        # Store instance params
        self.index = dates
        self.method = "ffill" if fillforward else None

        # Build dates
        self.dates = self.intersect(*dates)

    @staticmethod
    def intersect(*dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Get intersection set of all input dates
        :param dates: (pd.DatetimeIndex) input date indexes
        :return: (pd.DatetimeIndex) dates
        """

        _dates = dates[0]
        for d in dates[1:]:
            _dates = _dates.intersection(d)

        return _dates.sort_values()

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Place transformation logic here

        :param data: (pandas.DataFrame) input data to apply transformation to
        :param *other: (pandas.DataFrame) other data frames to use in transformation
        :return: (pandas.DataFrame) transformed data
        """

        # Re-index input dataframe to all days & fill forward so that there is
        # guaranteed to be data on all input dates
        all_days = pd.date_range(data.index.min(), data.index.max(), freq="D")
        data = data.reindex(all_days, method=self.method)

        # Finally re-index data back to union of all input dates
        return data.reindex(self.dates, method=self.method)

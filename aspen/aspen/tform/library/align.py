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

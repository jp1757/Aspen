"""
Objects for transforming asset returns data
"""
import pandas as pd

from aspen.tform.core import ITForm
from aspen.library.tform.align import Reindex


class Returns(ITForm):
    """
    Convert input total returns data to returns indexed to a
    specific set of dates
    """

    def __init__(self, dates: pd.DatetimeIndex):
        self.dates = dates
        self.reindex = Reindex(dates)

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns indexed to dates passed via __init__

        :param data: (pd.DataFrame) total returns indexed by date with assets as columns
        :param other: (pd.DataFrame) not used
        :return: (pd.DataFrame) returns data
        """

        # Re-index input data
        tr = self.reindex.apply(data)

        # Calculate returns
        return tr.pct_change()

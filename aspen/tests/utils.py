"""Utility to support tests"""

from typing import Tuple

import numpy as np
import pandas as pd


def returns(
        freq: str,
        *,
        sdate: pd.Timestamp = pd.Timestamp(year=2010, month=1, day=1),
        months: int = 12,
        stocks: list = ["aapl", "msft", "tsla", "vod"],
) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    """
    Generate random returns for 4 stocks over a period

    :param freq: (str) pandas frequency string
    :param sdate: (pd.TimeStamp) start date to build date range
    :param months: (int) number of months to add to start date
    :param stocks: (list) list of asset names
    :return: Tuple[pd.DatetimeIndex, pd.DataFrame] of dates built and returns
    """
    dates = pd.date_range(
        sdate,
        end=sdate + pd.DateOffset(months=months),
        freq=freq
    )
    _returns = pd.DataFrame(
        np.random.rand(len(dates), len(stocks)) / 20,
        index=pd.Index(dates, name="date"),
        columns=stocks
    )
    _returns.iloc[0] = 0

    return dates, _returns

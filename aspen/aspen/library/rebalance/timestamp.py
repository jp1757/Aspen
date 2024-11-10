"""
Rebalance implementations using logic based on the input date
"""

from typing import Set
import pandas as pd

from aspen.rebalance.core import IRebal


class DayOfWeek(IRebal):
    """Rebalance based on current day of week"""

    def __init__(self, days: Set) -> None:
        self.days = days

    def rebalance(
        self,
        *,
        date: pd.Timestamp,
        signals: pd.DataFrame,
        asset: pd.DataFrame,
    ) -> bool:
        return date.day_of_week in self.days


class BusinessMonthEnd(IRebal):

    def __init__(self) -> None:
        self.__date: pd.Timestamp = None
        self.__rebal_date: pd.Timestamp = None
        self.__counter = 0

    def rebalance(
        self,
        *,
        date: pd.Timestamp,
        signals: pd.DataFrame,
        asset: pd.DataFrame,
    ) -> bool:

        # First date received
        if self.__date is None:
            self.__date = date
            self.__rebal_date = date
            self.__counter = self.__counter + 1
            return True

        # Check if current date is last business day of month
        end_of_month = date == (date + pd.tseries.offsets.BMonthEnd(0))

        # Check if the month has changed & the previous date didn't trigger a
        # rebalance (trading holiday?)
        delayed_rebal = (date.month > self.__date.month) and (
            (self.__rebal_date.month < date.month - 1) or (self.__counter == 1)
        )

        # Update previous date flag
        self.__date = date

        rebal = delayed_rebal or end_of_month
        if rebal:
            self.__rebal_date = date
            self.__counter = self.__counter + 1

        return end_of_month or rebal

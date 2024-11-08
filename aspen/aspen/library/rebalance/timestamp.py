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

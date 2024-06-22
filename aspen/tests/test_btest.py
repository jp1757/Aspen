"""
Unit tests for backtest logic
"""

import unittest
import pandas as pd

from aspen.signals import ISignals
from aspen.pcr import IPortConstruct
from aspen.backtest.generic import BTest
import tests.utils


class Signals(ISignals):
    """
    Dummy signal combination object to be used with tests
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.data.dropna(how="all", inplace=True)

    def combine(self, normalise: bool) -> pd.DataFrame:
        return self.data


class PCR(IPortConstruct):
    """
    Dummy portfolio construction object to be used with tests
    """

    def weights(
            self, *, date: pd.Timestamp, signals: pd.DataFrame, asset: pd.DataFrame
    ) -> pd.Series:
        s = signals.iloc[-1]
        return s.divide(s.sum(), axis=0)


class TestBTest(unittest.TestCase):

    def setUp(self):
        # Load dummy returns
        self.dates, self.returns = tests.utils.returns(
            "B", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.tr = (1 + self.returns).cumprod()

        # Build dummy signal data
        m = self.returns.rolling(3).mean()
        sd = self.returns.rolling(3).std()
        self.sig_df = (self.returns - m) / sd

    def test_weights(self):
        """
        Test weights are calculated correctly for basic
        signal
        """

        # Create backtest object
        btest = BTest(
            dates=self.dates, tr=self.tr, signals=Signals(data=self.sig_df), pcr=PCR()
        )
        btdf = btest.run()

        # Run tests
        d1 = self.sig_df.index[0]
        wgt1 = PCR().weights(date=d1, signals=self.sig_df.iloc[[0]], asset=None)
        pd.testing.assert_series_equal(wgt1, btdf.loc[d1])

        d2 = self.sig_df.index[-1]
        wgt2 = PCR().weights(date=d2, signals=self.sig_df.iloc[[-1]], asset=None)
        pd.testing.assert_series_equal(wgt2, btdf.loc[d2])

        d3 = self.sig_df.index[10]
        wgt3 = PCR().weights(date=d3, signals=self.sig_df.iloc[[10]], asset=None)
        pd.testing.assert_series_equal(wgt3, btdf.loc[d3])

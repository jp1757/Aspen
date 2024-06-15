"""
Test asset returns calculations
"""

import unittest
import pandas as pd
import numpy as np

from aspen.tform.library.asset import Returns


class TestReturns(unittest.TestCase):

    @classmethod
    def returns(cls, freq: str):
        dates = pd.date_range(
            pd.Timestamp(year=2010, month=1, day=1),
            end=pd.Timestamp(year=2010, month=12, day=31),
            freq=freq
        )
        stocks = ["aapl", "msft", "tsla", "vod"]
        returns = pd.DataFrame(
            np.random.rand(len(dates), len(stocks)) / 20,
            index=pd.Index(dates, name="date"),
            columns=stocks
        )
        returns.iloc[0] = 0
        tr = (1 + returns).cumprod()

        return dates, tr

    def test_daily_returns(self):
        """
        Test business day returns are mapped to monthly dates
        """

        # Get dummy returns
        dates, tr = self.returns(freq="B")

        # Create target dates
        tgt_dates = pd.date_range(start=dates.min(), periods=12, freq="M")

        tf = Returns(tgt_dates)
        tfd = tf.apply(tr)

        self.assertTrue(tfd.loc[tgt_dates.min()].isna().all())
        self.assertFalse(tfd.iloc[1:].isna().any().any())
        self.assertEqual(len(tgt_dates), len(tfd))
        pd.testing.assert_series_equal(
            tr.loc["2010-02-26"] / tr.loc["2010-01-29"] - 1,
            tfd.loc["2010-02-28"],
            check_names=False
        )

    def test_bm_returns(self):
        """
        Test business month returns are mapped to calendar month dates
        """

        # Get dummy returns
        dates, tr = self.returns(freq="BM")

        # Create target dates
        tgt_dates = pd.date_range(start=dates.min(), periods=12, freq="M")

        tf = Returns(tgt_dates)
        tfd = tf.apply(tr)

        self.assertTrue(tfd.loc[tgt_dates.min()].isna().all())
        self.assertFalse(tfd.iloc[1:].isna().any().any())
        self.assertEqual(len(tgt_dates), len(tfd))
        pd.testing.assert_series_equal(
            tr.loc["2010-02-26"] / tr.loc["2010-01-29"] - 1,
            tfd.loc["2010-02-28"],
            check_names=False
        )
        pd.testing.assert_series_equal(
            tr.loc["2010-07-30"] / tr.loc["2010-06-30"] - 1,
            tfd.loc["2010-07-31"],
            check_names=False
        )

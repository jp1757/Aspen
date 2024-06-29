"""Unit tests for the stats package"""

import unittest

import numpy as np
import pandas as pd

import tests.utils
from aspen.stats import library as lib


class TestStats(unittest.TestCase):

    def setUp(self):
        # Load dummy data
        self.dBM, self.rBM = tests.utils.returns(
            "BM", sdate=pd.Timestamp(year=2010, month=1, day=1), months=26
        )

    def test_cagr(self):
        """Test compound average growth rate calculation"""

        # Test data
        tr = (1 + self.rBM.iloc[:, 0]).cumprod()

        # Declare test function
        def _cagr(*, tr: pd.Series, periods: int):
            return np.power(tr.iloc[-1] / tr.iloc[0], (periods / (len(tr) - 1))) - 1

        # Calculate CAGRs
        scalar = lib.cagr(tr=tr, periods=12)
        rolling = lib.cagr(tr=tr, periods=12, rolling=24)

        # Assertion statements
        self.assertEqual(scalar, _cagr(tr=tr, periods=12))
        self.assertEqual(rolling[0], _cagr(tr=tr[0:24], periods=12))
        self.assertEqual(rolling[1], _cagr(tr=tr[1:25], periods=12))
        self.assertEqual(rolling[2], _cagr(tr=tr[2:26], periods=12))

    def test_vol(self):
        """Test annualised volatility calculation"""

        # Test data
        tr = (1 + self.rBM.iloc[:, 0]).cumprod()

        # Declare test function
        def _vol(*, tr: pd.Series, periods: int):
            return np.std(tr.pct_change(), ddof=1) * np.sqrt(periods)

        # Calculate CAGRs
        scalar = lib.vol(tr=tr, periods=12)
        rolling = lib.vol(tr=tr, periods=12, rolling=24)

        # Assertion statements
        self.assertEqual(scalar, _vol(tr=tr, periods=12))
        self.assertEqual(rolling[0], _vol(tr=tr[0:24], periods=12))
        self.assertEqual(rolling[1], _vol(tr=tr[1:25], periods=12))
        self.assertEqual(rolling[2], _vol(tr=tr[2:26], periods=12))

    def test_excess(self):
        """Test calculating excess returns"""

        # Test data
        aapl = (1 + self.rBM.iloc[:, 0]).cumprod()
        msft = (1 + self.rBM.iloc[:, 1]).cumprod()

        # Calculate excess manually
        df = pd.concat([aapl, msft], axis=1).loc[:msft.index.max()]
        df["msft"] = df.msft.ffill()
        df = df.dropna(subset="aapl")
        df_xs = df.iloc[:, 0].pct_change() - df.iloc[:, 1].pct_change()

        # Use excess func
        xs = lib.excess(tr=aapl, other=msft)

        # Assertion statements
        pd.testing.assert_index_equal(xs.index, df_xs.index)
        pd.testing.assert_series_equal(xs, df_xs)

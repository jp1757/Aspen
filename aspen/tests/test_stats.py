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
        np.testing.assert_almost_equal(scalar, _cagr(tr=tr, periods=12))
        np.testing.assert_almost_equal(rolling[0], _cagr(tr=tr[0:24], periods=12))
        np.testing.assert_almost_equal(rolling[1], _cagr(tr=tr[1:25], periods=12))
        np.testing.assert_almost_equal(rolling[2], _cagr(tr=tr[2:26], periods=12))

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
        np.testing.assert_almost_equal(scalar, _vol(tr=tr, periods=12))
        np.testing.assert_almost_equal(rolling[0], _vol(tr=tr[0:24], periods=12))
        np.testing.assert_almost_equal(rolling[1], _vol(tr=tr[1:25], periods=12))
        np.testing.assert_almost_equal(rolling[2], _vol(tr=tr[2:26], periods=12))

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

    def test_sharpe(self):
        """Test sharpe ratio calculation"""

        # Test data
        tr = (1 + self.rBM.iloc[:, 0]).cumprod()

        # Declare test function
        def _sharpe(tr) -> float:
            cagr = np.power((tr.iloc[-1] / tr.iloc[0]), (12 / (len(tr) - 1))) - 1
            vol = np.std(tr.pct_change(), ddof=1) * np.sqrt(12)
            return cagr / vol

        # Calculate excess returns over rfr
        rate = 0.1
        rfr = np.power(rate + 1, 1 / 12) - 1
        df = pd.concat([tr, pd.Series(tr.pct_change(), name="ret")], axis=1)
        df["xs"] = df["ret"] - rfr
        df["xs"].iloc[0] = 0
        df["xs_tr"] = (1 + df["xs"]).cumprod()
        df["xs"].iloc[0] = np.NaN

        # Calculate CAGRs
        sharpe = lib.sharpe(tr=tr, periods=12)
        sharpe_xs = lib.sharpe(tr=tr, periods=12, rfr=rate)
        rolling = lib.sharpe(tr=tr, periods=12, rolling=24)

        # Assertion statements
        np.testing.assert_almost_equal(sharpe, _sharpe(tr=tr))
        np.testing.assert_almost_equal(sharpe_xs, _sharpe(tr=df["xs_tr"]))
        np.testing.assert_almost_equal(rolling[0], _sharpe(tr=tr[0:24]))
        np.testing.assert_almost_equal(rolling[1], _sharpe(tr=tr[1:25]))
        np.testing.assert_almost_equal(rolling[2], _sharpe(tr=tr[2:26]))

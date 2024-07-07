"""Unit tests for the stats package"""

import unittest

import numpy as np
import pandas as pd

import aspen.backtest.portfolio
import aspen.stats
import tests.utils


class TestStats(unittest.TestCase):

    def setUp(self):
        # Load dummy data
        self.dBM, self.rBM = tests.utils.returns(
            "BM", sdate=pd.Timestamp(year=2010, month=1, day=1), months=26
        )
        self.dW, self.rW = tests.utils.returns(
            "W", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )

    def test_cagr(self):
        """Test compound average growth rate calculation"""

        # Test data
        tr = (1 + self.rBM.iloc[:, 0]).cumprod()

        # Declare test function
        def _cagr(*, tr: pd.Series, periods: int):
            return np.power(tr.iloc[-1] / tr.iloc[0], (periods / (len(tr) - 1))) - 1

        # Calculate CAGRs
        scalar = tr.cagr(periods=12)
        rolling = tr.cagr(periods=12, rolling=24)

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
        scalar = tr.vol(periods=12)
        rolling = tr.vol(periods=12, rolling=24)

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
        xs = aapl.excess(other=msft)

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
        sharpe = tr.sharpe(periods=12)
        sharpe_xs = tr.sharpe(periods=12, rfr=rate)
        rolling = tr.sharpe(periods=12, rolling=24)

        # Assertion statements
        np.testing.assert_almost_equal(sharpe, _sharpe(tr=tr))
        np.testing.assert_almost_equal(sharpe_xs, _sharpe(tr=df["xs_tr"]))
        np.testing.assert_almost_equal(rolling[0], _sharpe(tr=tr[0:24]))
        np.testing.assert_almost_equal(rolling[1], _sharpe(tr=tr[1:25]))
        np.testing.assert_almost_equal(rolling[2], _sharpe(tr=tr[2:26]))

    def test_drawdown(self):
        """Test drawdown calulcations"""

        # Test data
        aapl = (1 + self.rBM.iloc[:, 0]).cumprod()
        msft = (1 + self.rBM.iloc[:, 1]).cumprod()

        ret = msft.pct_change() - aapl.pct_change()
        ret.iloc[0] = 0
        tr = (1 + ret).cumprod()
        tr.name = "tr"

        # Calculate drawdown dataframe
        rate = 0.01
        rfr = np.power(rate + 1, 1 / 12) - 1
        df = pd.concat([tr, pd.Series(tr.expanding().max(), name="max")], axis=1)
        df["dd"] = df["tr"] / df["max"] - 1
        df["xs"] = df["tr"].pct_change() - rfr
        df["xs"].iloc[0] = 0
        df["xs_tr"] = (1 + df["xs"]).cumprod()
        df["xs_max"] = df["xs_tr"].expanding().max()
        df["xs_dd"] = df["xs_tr"] / df["xs_max"] - 1

        # Assertion statements
        pd.testing.assert_series_equal(
            tr.drawdown(periods=12), df["dd"], check_names=False
        )
        pd.testing.assert_series_equal(
            tr.drawdown(periods=12, rfr=rate), df["xs_dd"], check_names=False
        )

    def test_turnover(self):
        """Test calculating drift turnover"""

        # Test data
        wgts = self.rBM.div(self.rBM.sum(axis=1), axis=0).dropna()
        tr = (1 + self.rBM).cumprod()

        # Portfolio object
        port = aspen.backtest.portfolio.Portfolio(asset_tr=tr, weights=wgts)

        # Calculate drifted weights
        drift_cols = [f"{x}_d" for x in port.asset_tr.columns]
        shift_cols = [f"{x}_d+1" for x in port.asset_tr.columns]

        drifted = port.drift(asset_tr=(1 + self.rW).cumprod())
        drift = drifted.rename(columns=dict(zip(drifted.columns, drift_cols)))

        df = pd.concat([port.weights, drift], axis=1)
        df[shift_cols] = df[drift_cols].shift(1)

        # Calculate turnover
        _s = df[shift_cols]
        _s.columns = port.asset_tr.columns
        diff = df[port.asset_tr.columns] - _s
        diff = diff.dropna().abs()
        turnover = diff.sum().sum() / (len(wgts) / 12)

        # Assertion statements
        np.testing.assert_almost_equal(wgts.turnover(periods=12, drifted=drifted), turnover)

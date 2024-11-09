"""Unit tests for the stats package"""

import sys
import os

import unittest
import pandas as pd
import numpy as np

import sklearn.linear_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import aspen.backtest.portfolio
import aspen.stats
import aspen.stats.library.signal
import utils


class TestPortfolio(unittest.TestCase):

    def setUp(self):
        # Load dummy data
        self.dBM, self.rBM = utils.returns(
            "BM", sdate=pd.Timestamp(year=2010, month=1, day=1), months=26
        )
        self.dW, self.rW = utils.returns(
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
        df = pd.concat([aapl, msft], axis=1).loc[: msft.index.max()]
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
        port = aspen.backtest.portfolio.Portfolio("test", asset_tr=tr, weights=wgts)

        # Calculate drifted weights
        drift_cols = [f"{x}_d" for x in port.asset_tr.columns]
        shift_cols = [f"{x}_d+1" for x in port.asset_tr.columns]

        drifted = aspen.backtest.portfolio.drift(
            asset_tr=(1 + self.rW).cumprod(), weights=wgts
        )
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
        np.testing.assert_almost_equal(
            wgts.turnover(periods=12, drifted=drifted), turnover
        )


class TestSignal(unittest.TestCase):

    def setUp(self):
        # Load dummy data
        self.dBM, self.rBM = utils.returns(
            "BM", sdate=pd.Timestamp(year=2010, month=1, day=1), months=26
        )
        self.dW, self.rW = utils.returns(
            "W", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.aapl = (1 + self.rBM.aapl).cumprod()

        self.tr = (1 + self.rBM).cumprod()
        self.zsc = {x: self.zscore(self.tr, x) for x in [3, 4, 5]}

    @staticmethod
    def zscore(tr: pd.Series, rolling: int) -> pd.Series:
        """Calculate zscore"""
        m = tr.rolling(rolling).mean()
        sd = tr.rolling(rolling).std()
        return ((tr - m) / sd).dropna()

    def test_ic(self):
        """Test information coefficient function"""

        zsc = self.zscore(self.aapl, 3)

        df = pd.concat([self.aapl, pd.Series(zsc, name="signal")], axis=1)
        df["ret"] = df["aapl"].pct_change()
        df["ret_3"] = df["aapl"].pct_change(3)
        df["ret+1"] = df["ret"].shift(-1)
        df["ret+3"] = df["ret_3"].shift(-3)

        np.testing.assert_almost_equal(
            df.iloc[1:5, 0].iloc[-1] / df.iloc[1:5, 0].iloc[0] - 1,
            df.iloc[len(df.iloc[1:5, 0]), 3],
        )

        sig_start = df["signal"].dropna().index[0]
        shift_3 = df.loc[sig_start:].iloc[0:4]
        np.testing.assert_almost_equal(
            shift_3.iloc[:, 0][-1] / shift_3.iloc[:, 0][0] - 1,
            df["ret+3"].loc[sig_start],
        )

        df.dropna(subset=["signal"], inplace=True)
        corr = df.corr()

        # Run functions to test
        ic_1, rolling_1 = aspen.stats.library.signal.ic(
            self.aapl, signal=zsc, lag=1, rolling=5
        )
        ic_3, rolling_3 = aspen.stats.library.signal.ic(
            self.aapl, signal=zsc, lag=3, rolling=5
        )

        # Assertion statements
        np.testing.assert_almost_equal(corr["signal"].loc["ret+1"], ic_1)
        np.testing.assert_almost_equal(corr["signal"].loc["ret+3"], ic_3)

        pd.testing.assert_series_equal(
            df[["signal", "ret+1"]]
            .rolling(5)
            .corr()
            .dropna()
            .iloc[:, 1]
            .loc[:, "signal"],
            rolling_1,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            df[["signal", "ret+3"]]
            .rolling(5)
            .corr()
            .dropna()
            .iloc[:, 1]
            .loc[:, "signal"],
            rolling_3,
            check_names=False,
        )

    def test_ic_xsect(self):
        """Test cross-sectionally information coefficient function"""

        tr = (1 + self.rBM).cumprod()
        signals = pd.concat([self.zscore(col, 5) for idx, col in tr.items()], axis=1)

        tr_cols = list(tr.columns)
        ret_cols = [f"{x}_ret_3" for x in tr_cols]
        ret3_cols = [f"{x}_ret+3" for x in tr_cols]
        sig_cols = [f"{x}_sig" for x in tr_cols]

        signal = signals.rename(columns=dict(zip(tr_cols, sig_cols)))
        df = pd.concat([tr, signal], axis=1)
        df[ret_cols] = df[tr_cols].pct_change(periods=3)
        df[ret3_cols] = df[ret_cols].shift(-3)
        df.dropna(subset=sig_cols, inplace=True, how="all")
        df.dropna(subset=ret3_cols, inplace=True, how="all")
        df = df[sig_cols + ret3_cols]

        def _series(rank):
            indexes = []
            values = []

            for index, row in df.iterrows():
                a = row.loc[sig_cols]
                a.index = [x.replace("_sig", "") for x in a.index]
                if rank:
                    a = a.rank()

                b = row.loc[ret3_cols]
                b.index = [x.replace("_ret+3", "") for x in b.index]

                indexes.append(index)
                values.append(a.corr(b))

            return pd.Series(data=values, index=indexes)

        # Calculate cross-sectional correlation
        xsect = aspen.stats.library.signal.ic_xsect(
            tr, signal=signals, lag=3
        )  # , rank="basic")
        xsect_rank = aspen.stats.library.signal.ic_xsect(
            tr, signal=signals, lag=3, rank="basic"
        )

        # Assertion statements
        pd.testing.assert_series_equal(
            _series(False), xsect, check_names=False, check_freq=False
        )
        pd.testing.assert_series_equal(
            _series(True), xsect_rank, check_names=False, check_freq=False
        )

    def test_pure_factor(self):
        """Test pure factor calculation"""

        # Calculate test data
        df = pd.concat(
            [pd.Series(self.tr.pct_change().shift(-1).stack(), name="ret")]
            + [pd.Series(z.stack(), name=f"zsc[{x}]") for x, z in self.zsc.items()],
            axis=1,
        )
        df = df.unstack().dropna().iloc[5].unstack().T

        x = df.iloc[:, 1:]
        y = df.iloc[:, 0]

        # Using sklearn
        regression = sklearn.linear_model.LinearRegression()
        regression.fit(x, y)

        # Run function
        fret, ftr = aspen.stats.library.signal.pure_factor(
            self.zsc[3],
            *[x for y, x in self.zsc.items() if y != 3],
            tr=self.tr,
            name="test",
        )

        np.testing.assert_almost_equal(regression.coef_[0], fret.iloc[6])

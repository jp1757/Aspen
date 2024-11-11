"""
Unit tests for backtest logic
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aspen.signals.generic import SignalDF, Signals, SignalsDF, Normalise
from aspen.pcr import IPortConstruct
from aspen.backtest.generic import BTest
import aspen.library.pcr.quintile
import aspen.backtest.portfolio
import aspen.library.tform.rank
import utils


class PCRProp(IPortConstruct):
    """
    Dummy portfolio construction object to be used with tests
    """

    def weights(
        self, *, date: pd.Timestamp, signals: pd.DataFrame, asset: pd.DataFrame
    ) -> pd.Series:
        s = signals.iloc[-1]
        return s.divide(s.sum(), axis=0)


class TestBTest(unittest.TestCase):
    """Test generic backtesting components"""

    def setUp(self):
        # Load dummy returns
        self.dates, self.returns = utils.returns(
            "B", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.tr = (1 + self.returns).cumprod()

        # Build dummy signal data
        m = self.returns.rolling(3).mean()
        sd = self.returns.rolling(3).std()
        self.sig_df = ((self.returns - m) / sd).dropna(how="all")

    def test_weights(self):
        """
        Test weights are calculated correctly for basic
        signal
        """

        # Create backtest object
        btest = BTest(
            "test",
            dates=self.dates,
            tr=self.tr,
            signals=SignalsDF(name="test", data=self.sig_df),
            pcr=PCRProp(),
        )
        btdf = btest.run()

        # Run tests
        d1 = self.sig_df.index[1]
        wgt1 = PCRProp().weights(date=d1, signals=self.sig_df.iloc[[0]], asset=None)
        pd.testing.assert_series_equal(wgt1, btdf.loc[d1], check_names=False)

        d2 = self.sig_df.index[-1]
        wgt2 = PCRProp().weights(date=d2, signals=self.sig_df.iloc[[-2]], asset=None)
        pd.testing.assert_series_equal(wgt2, btdf.loc[d2], check_names=False)

        d3 = self.sig_df.index[11]
        wgt3 = PCRProp().weights(date=d3, signals=self.sig_df.iloc[[10]], asset=None)
        pd.testing.assert_series_equal(wgt3, btdf.loc[d3], check_names=False)

    def test_quintile(self):
        """Test building quintile strategy weights"""

        # Test data
        dates = self.tr.index
        zsc = {x: utils.zsc(self.tr, x) for x in [3, 4]}
        signals = Signals(*[SignalDF(str(x), v) for x, v in zsc.items()], name="test")
        pcr = aspen.library.pcr.quintile.QuantileEW(long_bin=1, short_bin=3)
        normalise = Normalise(
            aspen.library.tform.rank.Quantile(
                rank=aspen.library.tform.rank.RankXSect(pct=False), bins=3
            )
        )

        # Build & run backtest object
        btest = BTest(
            "test",
            dates=dates,
            tr=self.tr,
            signals=signals,
            pcr=pcr,
            normalise=normalise,
            signal="3",
        )
        wgts = btest.run()
        summed = wgts.abs().sum(axis=1)

        # Assertion statements
        np.testing.assert_almost_equal(summed.iloc[1], 2.0)
        np.testing.assert_almost_equal(summed.iloc[5], 2.0)
        np.testing.assert_almost_equal(summed.iloc[7], 2.0)


class TestPortfolio(unittest.TestCase):
    """Test portfolio components"""

    def setUp(self):
        # Load dummy data
        self.dD, self.rD = utils.returns(
            "D", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.dB, self.rB = utils.returns(
            "B", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.dM, self.rM = utils.returns(
            "M", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.dBM, self.rBM = utils.returns(
            "BM", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.dBMoffset, self.rBMoffset = utils.returns(
            "BM", sdate=pd.Timestamp(year=2010, month=3, day=1), months=6
        )
        self.dW, self.rW = utils.returns(
            "W", sdate=pd.Timestamp(year=2010, month=3, day=1)
        )

    @staticmethod
    def returns(*, returns: pd.DataFrame, weights: pd.DataFrame) -> None:
        """
        Re-calculate returns using dummy weights and asset total returns data
        with varying frequency combinations

        :param returns: (pd.DataFrame) asset returns date as index, assets as columns
        :param weights: (pd.DataFrame) asset weights with date as index, assets as columns
        :return: None - runs assertion statements
        """

        # Prices
        tr = (1 + returns).cumprod()
        # Weights
        wgts = weights.divide(weights.sum(axis=1), axis=0).fillna(0)
        # Calculate returns with appropriate alignment
        tr_cols = [f"{x}_tr" for x in tr.columns]
        ret_cols = [f"{x}_ret" for x in tr_cols]
        ret_shift_cols = [f"{x}_-1" for x in ret_cols]
        wgt_cols = [f"{x}_wgt" for x in wgts.columns]
        df = pd.concat(
            [
                tr.rename(columns=dict(zip(tr.columns, tr_cols))),
                wgts.rename(columns=dict(zip(wgts.columns, wgt_cols))),
            ],
            axis=1,
        )
        df = df.ffill().loc[wgts.index]
        df[ret_cols] = (
            df[tr_cols].pct_change().rename(columns=dict(zip(tr.columns, ret_cols)))
        )
        df[ret_shift_cols] = df[ret_cols].shift(-1)
        w = df[wgt_cols].rename(
            columns=dict(zip(wgt_cols, [x.replace("_wgt", "") for x in wgt_cols]))
        )
        ret = df[ret_shift_cols].rename(
            columns=dict(
                zip(
                    ret_shift_cols,
                    [x.replace("_tr_ret_-1", "") for x in ret_shift_cols],
                )
            )
        )
        p_ret = w * ret
        p_ret = p_ret.shift(1)
        p_ret = p_ret.sum(axis=1, min_count=1)

        # Only leave one leading NaN
        start_idx = 0
        for x, (index, row) in enumerate(p_ret.items()):
            if not np.isnan(row):
                start_idx = x - 1
                break
        start_idx = max(start_idx, 0)
        p_ret = p_ret.iloc[start_idx:].copy()

        # Run returns function
        ret_act, tr_act, asset_act = aspen.backtest.portfolio.returns(
            dates=wgts.index, weights=wgts, asset_tr=tr
        )

        # Hack index frequency
        p_ret.index.freq = ret_act.index.freq

        # Assertion statements
        pd.testing.assert_series_equal(ret_act, p_ret)
        pd.testing.assert_series_equal(tr_act, (1 + ret_act.fillna(0)).cumprod())

    @staticmethod
    def drift(*, returns: pd.DataFrame, weights: pd.DataFrame) -> None:
        """
        Re-calculate drifted weights using dummy weights and asset total returns data
        with varying frequency combinations

        :param returns: (pd.DataFrame) asset returns date as index, assets as columns
        :param weights: (pd.DataFrame) asset weights with date as index, assets as columns
        :return: None - runs assertion statements
        """

        # Prices
        tr = (1 + returns).cumprod()
        # Weights
        wgts = weights.iloc[1:].divide(weights.iloc[1:].sum(axis=1), axis=0).fillna(0)

        # Column names
        tr_cols = [f"{x}_tr" for x in tr.columns]
        wgt_cols = [f"{x}_wgt" for x in tr.columns]
        ret_cols = [f"{x}_ret" for x in tr.columns]
        retshift_cols = [f"{x}_ret_-1" for x in tr.columns]
        pret_cols = [f"{x}_pret" for x in tr.columns]

        # Build master dataframe
        _tr = tr.rename(columns={x: y for x, y in zip(tr.columns, tr_cols)})
        _wgt = wgts.rename(columns={x: y for x, y in zip(tr.columns, wgt_cols)})
        df = pd.concat([_wgt, _tr], axis=1).ffill().dropna(how="all", subset=wgt_cols)
        df[ret_cols] = df[tr_cols].pct_change()
        df[retshift_cols] = df[ret_cols].shift(-1)
        df[pret_cols] = (
            df[wgt_cols]
            .rename(columns=dict(zip(wgt_cols, tr.columns)))
            .mul(df[retshift_cols].rename(columns=dict(zip(retshift_cols, tr.columns))))
        )
        df["pret_-1"] = df[pret_cols].sum(axis=1, min_count=1)

        # Calculate drifted weights
        _w = df[wgt_cols].rename(columns=dict(zip(wgt_cols, tr.columns)))
        _ra = df[retshift_cols].rename(columns=dict(zip(retshift_cols, tr.columns)))
        drift = (_w * (1 + _ra).div(1 + df["pret_-1"], axis=0)).shift(1)
        drift.loc[wgts.index] = np.NaN
        drift.dropna(how="all", inplace=True)
        drift = pd.concat([wgts, drift]).sort_index()

        # Run method to test
        port = aspen.backtest.portfolio.Portfolio("test", asset_tr=tr, weights=wgts)

        # Run assertion statements
        pd.testing.assert_frame_equal(
            aspen.backtest.portfolio.drift(asset_tr=tr, weights=wgts), drift
        )

    def test_monthly(self):
        """
        Test calculating portfolio returns from business month-end weights & calendar
        month-end asset total return data
        """

        self.returns(returns=self.rM, weights=self.rBM)

    def test_monthly_flip(self):
        """
        Test calculating portfolio returns from calendar month-end weights & business
        month-end asset total return data
        """

        self.returns(returns=self.rBM, weights=self.rM)

    def test_daily(self):
        """
        Test calculating portfolio returns from business day weights & calendar
        month-end asset total return data
        """

        self.returns(returns=self.rM, weights=self.rB)

    def test_daily_flip(self):
        """
        Test calculating portfolio returns from calendar month-end weights &
        business day weights asset total return data
        """

        self.returns(returns=self.rB, weights=self.rM)

    def test_monthly_offset(self):
        """
        Test calculating portfolio returns from business month-end weights & calendar
        month-end asset total return data where periods do not match
        """

        self.returns(returns=self.rM, weights=self.rBMoffset)

    def test_monthly_offset_flip(self):
        """
        Test calculating portfolio returns from calendar month-end weights & business
        month-end asset total return data where periods do not match
        """

        self.returns(returns=self.rBMoffset, weights=self.rM)

    def test_drift(self):
        """
        Test weights are drifted correctly using business month-end weights &
        business day asset total return prices
        """
        self.drift(returns=self.rB, weights=self.rBM)

    def test_drift_weekly(self):
        """
        Test weights are drifted correctly using business month-end weights &
        weekly asset total return prices
        """
        self.drift(returns=self.rW, weights=self.rBM)

    def test_btest(self):
        """Test end-to-end from signal to returns"""

        # Build dummy data
        svals = [
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
        ]
        sig = pd.Series(
            data=svals,
            index=pd.date_range(
                start="2000-01-07", end="2000-01-27", freq="B", name="date"
            ),
            name="asset",
        ).to_frame()

        trvals = [
            5106.4,
            5001.4,
            4939.3,
            4972.8,
            5057.3,
            4981.2,
            5000.2,
            4991.1,
            5093.1,
            5104.5,
            4957.2,
            4921.1,
            4844.2,
            4843.4,
            4872.3,
            4780.2,
            4879.9,
            4905.8,
        ]
        tr = pd.Series(
            data=trvals,
            index=pd.date_range(
                start="2000-01-04", end="2000-01-27", freq="B", name="date"
            ),
            name="asset",
        ).to_frame()

        # Define dummy portfolio construction object
        class _PCR(IPortConstruct):
            def weights(
                self, *, date: pd.Timestamp, signals: pd.DataFrame, asset: pd.DataFrame
            ) -> pd.Series:
                return signals.iloc[-1]

        # Build backtest object
        btest = BTest(
            name="test",
            dates=sig.index,
            tr=tr,
            signals=SignalsDF(name="test", data=sig),
            pcr=_PCR(),
        )

        # Build portfolio
        port = aspen.backtest.portfolio.Portfolio(
            name="test", asset_tr=tr, weights=btest.run()
        )

        # Test data
        tvals = pd.Series(
            data=[
                0,
                0,
                0.003814,
                0.001987,
                0.022464,
                0.024753,
                -0.004818,
                -0.012065,
                0.003373,
                0.003538,
                -0.00245,
                0.016407,
                -0.004792,
                -0.010074,
            ],
            index=pd.date_range(start="2000-01-10", end="2000-01-27", freq="B"),
        )

        # Assertion statement
        pd.testing.assert_series_equal(
            tvals, port.tr - 1, check_less_precise=6, check_names=False
        )

    def test_fx_adj(self):
        """Test fx_tr adjustment of asset total return prices"""

        # Test data
        atr = utils.returns(freq="M", stocks=["aapl", "msft", "vod", "bmw"])[1] + 1
        fx = (
            utils.returns(
                freq="B", sdate=pd.Timestamp(2010, 3, 31), stocks=["GBP", "EUR"]
            )[1]
            + 1
        )
        fx_map = {"aapl": "USD", "msft": "USD", "vod": "GBP", "bmw": "EUR"}

        # Call target function
        target_a = aspen.backtest.portfolio.fx_adjust(
            base="USD",
            base_denominated=True,
            dates=atr.index,
            asset_tr=atr,
            fx=fx,
            fx_map=fx_map,
        )
        target_b = aspen.backtest.portfolio.fx_adjust(
            base="USD",
            base_denominated=False,
            dates=atr.index,
            asset_tr=atr,
            fx=fx,
            fx_map=fx_map,
        )

        # Calculate test data
        fx["USD"] = 1
        fx = fx.reindex(atr.index, method="ffill")
        test_a = pd.concat(
            [pd.Series(y.div(fx[fx_map[x]]), name=x) for x, y in atr.items()], axis=1
        )
        test_b = pd.concat(
            [pd.Series(y.mul(fx[fx_map[x]]), name=x) for x, y in atr.items()], axis=1
        )

        # Run assertion statements
        pd.testing.assert_frame_equal(test_a, target_a)
        pd.testing.assert_frame_equal(test_b, target_b)

    def test_fx_port(self):
        """Test portfolio end-to-end with FX adjustment"""

        # Declare test data
        dts_td = pd.date_range(end=pd.Timestamp.now().date(), periods=5, name="date")
        fx_td = pd.DataFrame(
            zip(
                [0.7594, 0.7618, 0.7607, 0.7626, 0.7606],
                [0.9027, 0.9051, 0.9032, 0.9055, 0.9023],
            ),
            index=dts_td,
            columns=["GBP", "EUR"],
        )
        asset_td = pd.DataFrame(
            zip(
                [3725.6, 3721.3, 3697, 3723.4, 3747],
                [19126.51, 19127.52, 19142.66, 18969.05, 18798.48],
                [5671.04, 5722.59, 5722.59, 5602.04, 5590.16],
            ),
            index=dts_td,
            columns=["Z", "GX", "ES"],
        )
        wgts_td = pd.DataFrame(
            zip(
                [0.25, 0.2, 0.1, 0.25, 0.5],
                [0.35, 0.3, 0.5, 0.25, 0.15],
                [0.4, 0.5, 0.4, 0.5, 0.35],
            ),
            index=dts_td,
            columns=["Z", "GX", "ES"],
        )
        fx_map_td = {"Z": "GBP", "GX": "EUR", "ES": "USD"}

        # Init portfolio objects - 1st base currency denominated USD/x,
        # 2nd other way around x/USD
        port = aspen.backtest.portfolio.Portfolio(
            "test",
            asset_tr=asset_td,
            weights=wgts_td,
            fx="USD",
            base_denominated=True,
            fx_tr=fx_td,
            fx_map=fx_map_td,
        )
        port_nonb = aspen.backtest.portfolio.Portfolio(
            "test",
            asset_tr=asset_td,
            weights=wgts_td,
            fx="USD",
            base_denominated=False,
            fx_tr=1 / fx_td,
            fx_map=fx_map_td,
        )

        # Target test data
        target = [
            0.00165113294204722,
            -0.000149628531359913,
            -0.0137562235269713,
            -0.000183565827636567,
        ]

        # Assertion statements
        np.testing.assert_array_almost_equal(port.returns.dropna().values, target)
        np.testing.assert_array_almost_equal(port_nonb.returns.dropna().values, target)

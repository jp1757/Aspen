"""
Unit tests for backtest logic
"""

import unittest
import pandas as pd
import numpy as np

from aspen.signals import ISignals
from aspen.signals.generic import SignalDF, Signals
from aspen.pcr import IPortConstruct
from aspen.backtest.generic import BTest
import aspen.pcr.library.quintile
import aspen.backtest.portfolio
import aspen.signals.library.normalise
import tests.utils


class SignalsDummy(ISignals):
    """
    Dummy signal combination object to be used with tests
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.data.dropna(how="all", inplace=True)

    def build(self, name: str = None) -> pd.DataFrame:
        """
        Serve up signal data by either combining multiple signals or
        returning a specific signal by setting the 'name' parameter

        :param name: (str, optional) name of signal to return
        :return: pd.DataFrame of signal data indexed by date with columns set
            to asset ids
        """
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
    """Test generic backtesting components"""

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
            dates=self.dates, tr=self.tr, signals=SignalsDummy(data=self.sig_df), pcr=PCR()
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

    def test_quintile(self):
        """Test building quintile strategy weights"""

        # Test data
        dates = self.tr.index
        zsc = {x: tests.utils.zsc(self.tr, x) for x in [3, 4]}
        signals = Signals(*[SignalDF(str(x), v) for x, v in zsc.items()])
        pcr = aspen.pcr.library.quintile.QuantileEW(long_bin=1, short_bin=3)
        normalise = aspen.signals.library.normalise.Quantile(
            rank=aspen.tform.library.rank.RankXSect(pct=False), bins=3
        )

        # Build & run backtest object
        btest = BTest(
            dates=dates,
            tr=self.tr,
            signals=signals,
            pcr=pcr,
            normalise=normalise,
            signal="3"
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
        self.dD, self.rD = tests.utils.returns(
            "D", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.dB, self.rB = tests.utils.returns(
            "B", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.dM, self.rM = tests.utils.returns(
            "M", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.dBM, self.rBM = tests.utils.returns(
            "BM", sdate=pd.Timestamp(year=2010, month=1, day=1)
        )
        self.dBMoffset, self.rBMoffset = tests.utils.returns(
            "BM", sdate=pd.Timestamp(year=2010, month=3, day=1), months=6
        )
        self.dW, self.rW = tests.utils.returns(
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
                wgts.rename(columns=dict(zip(wgts.columns, wgt_cols)))
            ],
            axis=1
        )
        df = df.ffill().loc[wgts.index]
        df[ret_cols] = df[tr_cols].pct_change().rename(
            columns=dict(zip(tr.columns, ret_cols))
        )
        df[ret_shift_cols] = df[ret_cols].shift(-1)
        w = df[wgt_cols].rename(
            columns=dict(zip(wgt_cols, [x.replace("_wgt", "") for x in wgt_cols]))
        )
        ret = df[ret_shift_cols].rename(
            columns=dict(
                zip(ret_shift_cols, [x.replace("_tr_ret_-1", "") for x in ret_shift_cols])
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
        ret_act, tr_act = aspen.backtest.portfolio.returns(
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
        df[pret_cols] = df[wgt_cols].rename(columns=dict(zip(wgt_cols, tr.columns))).mul(
            df[retshift_cols].rename(columns=dict(zip(retshift_cols, tr.columns)))
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
        port = aspen.backtest.portfolio.Portfolio(asset_tr=tr, weights=wgts)

        # Run assertion statements
        pd.testing.assert_frame_equal(port.drift(tr), drift)

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

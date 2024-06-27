"""
Unit tests for backtest logic
"""

import unittest
import pandas as pd
import numpy as np

from aspen.signals import ISignals
from aspen.pcr import IPortConstruct
from aspen.backtest.generic import BTest
import aspen.backtest.portfolio
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

    @staticmethod
    def returns(*, returns: pd.DataFrame, weights: pd.DataFrame) -> None:
        """
        Re-calculate returns using dummy weights and asset total returns data
        with varying frequency combinations

        :param returns: (pd.DataFrame) asset returns date as index, assets as columns
        :param weights: (pd.DataFrame) asset weights with date as index, assets as columns
        :return: None - runs assertion statements
        """

        # Prices M
        tr = (1 + returns).cumprod()
        # Weights BM
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

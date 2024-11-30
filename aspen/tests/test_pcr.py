"""
Test portfolio construction implementations
"""

import sys
import os

import unittest
import pandas as pd
import numpy as np
import cvxpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aspen.library.pcr.quantile import QuantileEW
import aspen.library.pcr.mvo
import aspen.tform.generic
import utils


class Quantile(unittest.TestCase):
    """
    Test class & functions in quintile module
    """

    def test_quantile_ew(self):
        """Test quantile ew object calculates weights from bins correctly"""

        # Test signal data - typically binned data
        bins = pd.DataFrame(
            [[1, 3, 3, 1, 2, 2, 1], [1, 1, 2, 2, 3, 3, 1]],
            columns=["aapl", "msft", "tsla", "vod", "eon", "meta", "amzn"],
            index=pd.DatetimeIndex(
                ["2010-03-31", "2010-04-30"], dtype="datetime64[ns]", name="date"
            ),
        )

        # Init pcr object
        qew = QuantileEW(long_bin=1, short_bin=3)

        # Get weights (which also runs built-in assertion statements)
        wgts = qew.weights(date=pd.Timestamp("2010-04-30"), signals=bins, asset=None)

        # Run assertion statements
        np.testing.assert_almost_equal(wgts.loc["tsla"], 0.0)
        np.testing.assert_almost_equal(wgts.loc["msft"], 0.333, decimal=3)
        np.testing.assert_almost_equal(wgts.loc["meta"], -0.5)


class MVO(unittest.TestCase):
    """Test mean variance optimization class & functions"""

    def test_vcv(self):
        """Test vcv function"""

        # Init weekly & monthly dates
        dts_w = pd.date_range(start="2024-01-01", freq="W", periods=52)
        dts_m = pd.date_range(start="2024-01-01", end=dts_w.max(), freq="M")

        # Dummy asset returns
        td_dts, td = utils.returns(freq="B", sdate=dts_w.min())
        td_tr = (1 + td.loc[: dts_w.max()]).cumprod()

        # Re-index to monthly
        td_m = (
            pd.concat([td_tr, pd.Series(np.NaN, index=dts_m, name="Months")], axis=1)
            .ffill()
            .loc[dts_m, td_tr.columns]
        )
        # Only calculate last 2 VCVs
        td_m_ret = td_m.pct_change().tail(4)
        cov_1 = td_m_ret.head(3).cov()
        cov_2 = td_m_ret.tail(3).cov()

        # Call target func
        vcv_t = aspen.library.pcr.mvo.vcv(
            dates=dts_w,
            asset_tr=td_tr,
            freq="M",
            window=aspen.tform.generic.TForm("rolling", window=3),
        )

        for x in dts_w[-9:-5]:
            pd.testing.assert_frame_equal(cov_1, vcv_t.loc[x])

        for x in dts_w[-5:]:
            pd.testing.assert_frame_equal(cov_2, vcv_t.loc[x])

    def test_fixed_risk(self):
        """
        Test PCR class - tests that it just doesn't through an error more than anything...
        """

        # Test dates
        dts_w = pd.date_range(start="2024-01-01", freq="W", periods=52)
        # Dummy asset data
        td_dts, td = utils.returns(freq="B", sdate=dts_w.min())
        td_tr = (1 + td.loc[: dts_w.max()]).cumprod()
        td_ret = td_tr.pct_change()
        # Dummy Signal
        s = td_ret.reindex(dts_w, method="ffill").loc["2024-12-22"]
        s = ((s - s.mean()) / s.std()).to_frame().T

        # VCV
        vcv_t = aspen.library.pcr.mvo.vcv(
            dates=dts_w,
            asset_tr=td_tr,
            freq="M",
            window=aspen.tform.generic.TForm("rolling", window=3),
        )
        cov = aspen.library.pcr.mvo.zero_correlations(vcv_t.loc[s.index[0]])

        # Convert signal to alpha
        vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.columns)
        alpha = (0.1 * s * vol).iloc[0]

        # Run optimization
        opt_weights = aspen.library.pcr.mvo.optimize(
            cov=cov,
            risk_aversion=10,
            alpha=alpha,
            constraints=None,
            solver=cvxpy.ECOS,
        )

        # Call target & assert
        pd.testing.assert_series_equal(
            aspen.library.pcr.mvo.FixedRisk(cov=vcv_t, risk_aversion=10).weights(
                date=s.index[0], signals=s, asset=pd.DataFrame()
            ),
            pd.Series(opt_weights.value, index=alpha.index),
            check_names=False,
        )

"""
Test asset returns calculations
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
)

from aspen.library.tform.asset import Returns
import utils


class TestReturns(unittest.TestCase):

    def test_daily_returns(self):
        """
        Test business day returns are mapped to monthly dates
        """

        # Get dummy returns
        dates, tr = utils.returns(freq="B")

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
        dates, tr = utils.returns(freq="BM")

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

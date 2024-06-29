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
       
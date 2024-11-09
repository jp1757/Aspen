"""Unit tests for some of the IRebal implementations"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import aspen.library.rebalance.timestamp


class TestTimestamp(unittest.TestCase):

    def test_bm(self):
        """Test business month-end implementation"""

        # Test 1 - standard end of month
        rebal = aspen.library.rebalance.timestamp.BusinessMonthEnd()
        self.assertListEqual(
            [
                rebal.rebalance(date=x, signals=None, asset=None)
                for x in pd.date_range("2000-01-27", periods=7, freq="B")
            ],
            [True, False, True, False, False, False, False],
        )

        # Test 2 -
        rebal = aspen.library.rebalance.timestamp.BusinessMonthEnd()
        self.assertListEqual(
            [
                rebal.rebalance(date=x, signals=None, asset=None)
                for x in pd.date_range("2000-04-25", periods=7, freq="B")
            ],
            [True, False, False, True, False, False, False],
        )

        # Test 3 - end of month is skipped so rebalance start of next
        rebal = aspen.library.rebalance.timestamp.BusinessMonthEnd()
        dts = pd.date_range("2000-04-25", periods=7, freq="B")
        dts = dts[0:3].union(dts[4:])
        self.assertListEqual(
            [rebal.rebalance(date=x, signals=None, asset=None) for x in dts],
            [True, False, False, True, False, False],
        )

        # Test 4 - end of month & start of month skipped
        rebal = aspen.library.rebalance.timestamp.BusinessMonthEnd()
        dts = pd.date_range("2000-04-25", periods=7, freq="B")
        dts = dts[0:3].union(dts[6:])
        self.assertListEqual(
            [rebal.rebalance(date=x, signals=None, asset=None) for x in dts],
            [True, False, False, True],
        )

        # Test 5 - end of month & start of month skipped then next month end of month
        rebal = aspen.library.rebalance.timestamp.BusinessMonthEnd()
        dts = pd.date_range("2000-04-25", periods=28, freq="B")
        dts = dts[0:3].union(dts[6:])
        self.assertListEqual(
            [rebal.rebalance(date=x, signals=None, asset=None) for x in dts],
            [
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
            ],
        )

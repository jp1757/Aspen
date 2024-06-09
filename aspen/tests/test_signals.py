"""Unit tests for Signals Package"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aspen.tform import TForm, Merge, Pipeline
from aspen.signals import Signal, Leaf, LeafSeries


class TestSignal(unittest.TestCase):

    def setUp(self):
        # Create random returns for 4 stocks
        dates = pd.date_range("01/01/2020", periods=12, freq="BME")
        stocks = ["aapl", "msft", "tsla", "vod"]
        returns = pd.DataFrame(
            np.random.rand(len(dates), len(stocks)),
            index=pd.Index(dates, name="date"),
            columns=stocks
        )
        returns.iloc[0] = 0

        # Create random fundamentals
        net_income = pd.DataFrame(
            np.random.rand(len(dates), len(stocks)),
            index=pd.Index(dates, name="date"),
            columns=stocks
        ) * 1e6
        shares = pd.DataFrame(
            np.random.rand(len(dates), len(stocks)),
            index=pd.Index(dates, name="date"),
            columns=stocks
        ) * 1e7

        # Data heap
        self.data = {
            "returns": returns,
            "net_income": net_income,
            "shares": shares
        }

    def test_signal_leaves(self):
        """
        Test a basic signal is calculated correctly

        Calculate a dummy price/earnings ratio from returns, net income & shares outstanding
        """

        # Build signal object passing various transformation objects
        sig = Signal(
            Leaf(
                Pipeline(TForm("add", other=1), TForm("cumprod")),
                "tr",
                ["returns"]
            ),
            Leaf(Merge("div"), "eps", ["net_income", "shares"]),
            Leaf(Merge("div"), "pe", ["tr", "eps"]),
            LeafSeries(Pipeline(TForm("rolling", window=3), TForm("mean")), "signal"),
            data=self.data
        )
        signal = sig.calculate()

        # Calculate test data
        tr = (self.data["returns"] + 1).cumprod()
        eps = self.data["net_income"] / self.data["shares"]
        pe = tr / eps
        rolling_pe = pe.rolling(3).mean()

        pd.testing.assert_frame_equal(signal, rolling_pe)

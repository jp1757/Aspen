"""Unit tests for SignalsDummy Package"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aspen.tform.generic import TForm, Merge
from aspen.tform.pipeline import Pipeline
from aspen.signals.core import ISignal
from aspen.signals.generic import SignalHeap
from aspen.signals.leaf import Leaf, LeafSeries
from aspen.library.signals.signals import SMean


class DummySignal(ISignal):

    def __init__(self, name: str, data: pd.DataFrame) -> None:
        self._name = name
        self.data = data

    @property
    def name(self) -> str:
        return self._name

    def calculate(self) -> pd.DataFrame:
        """
        Calculate a signal object & return data
        :return: pd.DataFrame
        """
        return self.data


class TestSignal(unittest.TestCase):

    def setUp(self):
        # Create random returns for 4 stocks
        dates = pd.date_range("01/01/2020", periods=12, freq="BM")
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
        sig = SignalHeap(
            Leaf(
                Pipeline(TForm("add", other=1), TForm("cumprod")),
                "asset_tr",
                ["returns"]
            ),
            Leaf(Merge("div"), "eps", ["net_income", "shares"]),
            Leaf(Merge("div"), "pe", ["asset_tr", "eps"]),
            LeafSeries(Pipeline(TForm("rolling", window=3), TForm("mean")), "signal"),
            data=self.data,
            name="test"
        )
        signal = sig.calculate()

        # Calculate test data
        tr = (self.data["returns"] + 1).cumprod()
        eps = self.data["net_income"] / self.data["shares"]
        pe = tr / eps
        rolling_pe = pe.rolling(3).mean()

        pd.testing.assert_frame_equal(signal, rolling_pe)

    def test_invalid_mapping(self):
        """Test mapping not found raises exception"""

        sig = SignalHeap(
            Leaf(
                Pipeline(TForm("rolling", window=3), TForm("mean")),
                "signal",
                mappings=["asset_tr", "null"]
            ),
            data={"asset_tr": (1 + self.data["returns"]).cumprod()},
            name="test"
        )

        with self.assertRaises(Exception) as context:
            sig.calculate()

        print(str(context))

        self.assertTrue("Mappings not found in heap: {'null'}" in str(context.exception))


class TestSignals(unittest.TestCase):
    """
    Test ISignals implementations
    """

    def setUp(self):
        self.bins = pd.DataFrame(
            [[1, 3, 3, 1, 2, 2, 1], [1, 1, 2, 2, 3, 3, 1]],
            columns=['aapl', 'msft', 'tsla', 'vod', 'eon', 'meta', 'amzn'],
            index=pd.DatetimeIndex(
                ['2010-03-31', '2010-04-30'], dtype='datetime64[ns]', name='date'
            )
        )

    def test_smean(self):
        """Test SMean object"""
        smean = SMean(
            DummySignal("bins1", self.bins), DummySignal("bins2", self.bins + 1)
        )
        build = smean.build()

        pd.testing.assert_series_equal(
            build.iloc[0].sort_index(),
            ((self.bins.iloc[0] + self.bins.iloc[0] + 1) / 2).sort_index()
        )
        pd.testing.assert_frame_equal(self.bins, smean.build("bins1"))

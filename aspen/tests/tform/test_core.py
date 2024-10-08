"""Unit tests for TForms Package"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
)

import aspen.tform.generic
from aspen.tform import TForm, Merge, Pipeline


class TestTForm(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(
            [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 5]], columns=["a", "b", "c", "d"]
        )

    def test_call(self):
        """
        Test calling a transform on the data object, where the dataframe calls the func
        """
        tf = TForm("sum")
        data = tf.apply(self.data)

        pd.testing.assert_series_equal(self.data.sum(), data)

    def test_pass(self):
        """
        Test calling a transform where the dataframe is passed to the target func
        """
        tf = TForm("sum", mode=aspen.tform.generic.Mode.PASS, lib="numpy")
        data = tf.apply(self.data)

        pd.testing.assert_series_equal(np.sum(self.data), data)

    def test_call_params(self):
        """
        Test calling a transform on the data object, where the dataframe calls the func.
        Passing extra keyword args to func
        """
        tf = TForm("sum", axis=1)
        data = tf.apply(self.data)

        pd.testing.assert_series_equal(self.data.sum(axis=1), data)

    def test_pass_params(self):
        """
        Test calling a transform where the dataframe is passed to the target func.
        Passing extra keyword args to func
        """
        tf = TForm("sum", mode=aspen.tform.generic.Mode.PASS, lib="numpy", axis=1)
        data = tf.apply(self.data)

        pd.testing.assert_series_equal(np.sum(self.data, axis=1), data)


class TestMerge(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(
            [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 5]], columns=["a", "b", "c", "d"]
        )
        self.data2 = self.data.copy()
        self.data2.iloc[0, 1] = np.NaN

    def test_call(self):
        """
        Test calling a transform on the data object, where the dataframe calls the func
        """
        data = Merge("mul").apply(self.data, self.data2)

        pd.testing.assert_frame_equal(self.data.mul(self.data2), data)

    def test_pass(self):
        """
        Test calling a transform where the dataframe is passed to the target func
        """
        tf = Merge("multiply", mode=aspen.tform.generic.Mode.PASS, lib="numpy")
        data = tf.apply(self.data, self.data2)

        pd.testing.assert_frame_equal(np.multiply(self.data, self.data2), data)

    def test_call_params(self):
        """
        Test calling a transform on the data object, where the dataframe calls the func.
        Passing extra keyword args to func
        """
        data = Merge("mul", fill_value=0).apply(self.data, self.data2)

        pd.testing.assert_frame_equal(self.data.mul(self.data2, fill_value=0), data)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 5]], columns=["a", "b", "c", "d"]
        )

    def test_pipeline_call(self):
        """Test basic pipeline"""

        pl = Pipeline(TForm("sum"), TForm("mean"))
        self.assertEqual(pl.apply(self.data), 8.0)

    def test_pipeline_pass(self):
        """Test basic pipeline with pass TForms"""
        pl = Pipeline(
            TForm("sum", mode=aspen.tform.generic.Mode.PASS, lib="numpy"),
            TForm("mean", mode=aspen.tform.generic.Mode.PASS, lib="numpy")
        )
        self.assertEqual(pl.apply(self.data), 8.0)

    def test_diagnostics(self):
        """Test diagnostics saved correctly"""
        pl = Pipeline(TForm("sum"), TForm("mean"), save=True)
        res = pl.apply(self.data)

        self.assertEqual(len(pl.diagnostics), 3)

        pd.testing.assert_frame_equal(pl.diagnostics[0][1], self.data)
        self.assertEqual(pl.diagnostics[-1][1], res)

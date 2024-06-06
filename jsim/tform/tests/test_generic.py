"""Unit tests for Generic TForm"""

import unittest

import pandas as pd
import numpy as np

import tform.generic
from tform import Generic


class TestGeneric(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(
            [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 5]], columns=["a", "b", "c", "d"]
        )

    def test_call(self):
        """
        Test calling a transform on the data object, where the dataframe calls the func
        """
        tf = Generic(tform.generic.Mode.CALL, "sum")
        data = tf.apply(self.data)

        pd.testing.assert_series_equal(self.data.sum(), data)

    def test_pass(self):
        """
        Test calling a transform where the dataframe is passed to the target func
        """
        tf = Generic(tform.generic.Mode.PASS, "sum", lib="numpy")
        data = tf.apply(self.data)

        pd.testing.assert_series_equal(np.sum(self.data), data)

    def test_call_params(self):
        """
        Test calling a transform on the data object, where the dataframe calls the func.
        Passing extra keyword args to func
        """
        tf = Generic(tform.generic.Mode.CALL, "sum", axis=1)
        data = tf.apply(self.data)

        pd.testing.assert_series_equal(self.data.sum(axis=1), data)

    def test_pass_params(self):
        """
        Test calling a transform where the dataframe is passed to the target func.
        Passing extra keyword args to func
        """
        tf = Generic(tform.generic.Mode.PASS, "sum", lib="numpy", axis=1)
        data = tf.apply(self.data)

        pd.testing.assert_series_equal(np.sum(self.data, axis=1), data)

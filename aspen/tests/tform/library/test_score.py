"""Test score transformation implementations"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import aspen.library.tform.score
import utils


class TestScore(unittest.TestCase):

    def test_zsc_xs(self):
        """Test cross-setional score"""

        # Create random test data
        td = pd.DataFrame(np.random.uniform(-10, 10, (10, 4)))

        # Z-SCore calculation
        def _zsc(s: pd.Series):
            _m = s.mean()
            _sd = s.std()
            return (s - _m) / _sd

        # Loop through td rows and calculate z-scores
        lazy = [_zsc(y) for x, y in td.iterrows()]
        lazy_df = pd.DataFrame(lazy)

        # Get target values
        tf = aspen.library.tform.score.ZScoreXS()
        tf_df = tf.apply(td)

        # Run assertion statement
        np.testing.assert_array_almost_equal(tf_df.values, lazy_df.values)

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
import aspen.tform.generic
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

    def test_zsc(self):
        """Test timeseries z-score calculations by column"""

        # Create random test data
        td = pd.DataFrame(np.random.uniform(-10, 10, (30, 3)))

        # Assert rolling
        rolling = {
            x: (y - y.rolling(window=5, min_periods=5).mean())
            / y.rolling(window=5, min_periods=5).std()
            for x, y in td.items()
        }
        rolling_tf = aspen.library.tform.score.ZScore(
            aspen.tform.generic.TForm("rolling", window=5, min_periods=5)
        )
        rolling_df = rolling_tf.apply(td)
        pd.testing.assert_frame_equal(pd.DataFrame(rolling), rolling_df)
        pd.testing.assert_series_equal(
            (td.iloc[4] - td.head(5).mean()) / td.head(5).std(),
            rolling_df.iloc[4],
            check_names=False,
        )

        # Calculate for each column independently
        ewm = {
            x: (y - y.ewm(halflife=5, min_periods=5).mean())
            / y.ewm(halflife=5, min_periods=5).std()
            for x, y in td.items()
        }
        ewm_tf = aspen.library.tform.score.ZScore(
            aspen.tform.generic.TForm("ewm", halflife=5, min_periods=5)
        )
        pd.testing.assert_frame_equal(pd.DataFrame(ewm), ewm_tf.apply(td))

    def test_xsweights(self):
        """Test cross-sectional weights calculation"""

        # Random scores
        scores = pd.DataFrame(
            np.random.uniform(-2, 2, (10, 4)),
            columns=["a", "b", "c", "d"],
            index=pd.date_range(end=pd.Timestamp.now().date(), periods=10, name="date"),
        )

        # Init normalise obj
        norm = aspen.library.tform.score.XSWeights()
        norm_df = norm.apply(scores)

        # Assertion statements
        np.testing.assert_array_almost_equal(norm_df.abs().sum(axis=1), 1.0)
        pd.testing.assert_frame_equal(norm_df < 0, scores < 0)
        pd.testing.assert_frame_equal(norm_df > 0, scores > 0)

"""Test implementation of INormalise"""

import unittest
import numpy as np
import pandas as pd

import aspen.library.signals.normalise


class TestNormalise(unittest.TestCase):

    def test_xsweights(self):
        """Test cross-sectional weights calculation"""

        # Random scores
        scores = pd.DataFrame(
            np.random.uniform(-2, 2, (10, 4)),
            columns=["a", "b", "c", "d"],
            index=pd.date_range(end=pd.Timestamp.now().date(), periods=10, name="date"),
        )

        # Init normalise obj
        norm = aspen.library.signals.normalise.XSWeights()
        norm_df = norm.norm(scores)

        # Assertion statements
        np.testing.assert_array_almost_equal(norm_df.abs().sum(axis=1), 1.0)
        pd.testing.assert_frame_equal(norm_df < 0, scores < 0)
        pd.testing.assert_frame_equal(norm_df > 0, scores > 0)

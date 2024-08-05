"""
Test portfolio construction implementations
"""

import unittest
import pandas as pd
import numpy as np

from aspen.library.pcr.quintile import QuantileEW


class TestLibrary(unittest.TestCase):
    """
    Test pcr objects in library package
    """

    def test_quantile_ew(self):
        """Test quantile ew object calculates weights from bins correctly"""

        # Test signal data - typically binned data
        bins = pd.DataFrame(
            [[1, 3, 3, 1, 2, 2, 1], [1, 1, 2, 2, 3, 3, 1]],
            columns=['aapl', 'msft', 'tsla', 'vod', 'eon', 'meta', 'amzn'],
            index=pd.DatetimeIndex(
                ['2010-03-31', '2010-04-30'], dtype='datetime64[ns]', name='date'
            )
        )

        # Init pcr object
        qew = QuantileEW(long_bin=1, short_bin=3)

        # Get weights (which also runs built-in assertion statements)
        wgts = qew.weights(
            date=pd.Timestamp('2010-04-30'), signals=bins, asset=None
        )

        # Run assertion statements
        np.testing.assert_almost_equal(wgts.loc["tsla"], 0.0)
        np.testing.assert_almost_equal(wgts.loc["msft"], 0.333, decimal=3)
        np.testing.assert_almost_equal(wgts.loc["meta"], -0.5)

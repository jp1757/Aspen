"""
Test alignment transformations
"""

import unittest
import pandas as pd

from aspen.tform.library.align import Reindex, Align
import tests.utils as utils


class TestAlign(unittest.TestCase):

    def test_reindex(self):
        """Test reindex dates are all found in transformed dataframe"""
        dates, returns = utils.returns("B")
        months = pd.date_range(dates.min(), periods=10, freq="M")

        data = Reindex(months).apply(returns)

        self.assertEqual(len(data), len(months))
        self.assertFalse(data.isna().any().any())

    def test_align(self):
        """Test the alignment of multiple dataframes"""

        # Build dummy test data
        d1, ret1 = utils.returns("B", sdate=pd.Timestamp(year=2010, month=1, day=1))
        d2, ret2 = utils.returns("D", sdate=pd.Timestamp(year=2010, month=5, day=1))
        d3, ret3 = utils.returns("BM", sdate=pd.Timestamp(year=2010, month=1, day=1), months=9)
        dates = d1.intersection(d2).intersection(d3)

        # Create TForm
        tf = Align(ret1.index, ret2.index, ret3.index)
        # Apply
        dfs = [tf.apply(x) for x in [ret1, ret2, ret3]]

        # Assert
        for df in dfs:
            self.assertEqual(len(df), len(dates))
            self.assertFalse(df.isna().any().any())

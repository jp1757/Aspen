"""
Test alignment transformations
"""

import unittest
import pandas as pd

from aspen.tform.library.align import Reindex
import tests.tform.library.utils as utils


class TestReindex(unittest.TestCase):

    def test_months(self):
        """Test reindex dates are all found in transformed dataframe"""
        dates, returns = utils.returns("B")
        months = pd.date_range(dates.min(), periods=10, freq="M")

        data = Reindex(months).apply(returns)

        self.assertEqual(len(data), len(months))

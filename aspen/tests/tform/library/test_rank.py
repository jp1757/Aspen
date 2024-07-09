"""
Test ranking transformations
"""

import unittest
from typing import Union
import pandas as pd

import tests.utils as utils
import aspen.tform.library.rank


class TestRank(unittest.TestCase):

    def setUp(self):
        stocks = ["aapl", "msft", "tsla", "vod", "eon", "meta", "amzn"]
        self.dBM, self.rBM = utils.returns(
            "BM", sdate=pd.Timestamp(year=2010, month=1, day=1), months=26, stocks=stocks
        )
        self.tr = (1 + self.rBM).cumprod()
        self.zsc = self.zscore(self.tr, 4)

    @staticmethod
    def zscore(
            tr: Union[pd.Series, pd.DataFrame], rolling: int
    ) -> Union[pd.Series, pd.DataFrame]:
        m = tr.rolling(rolling).mean()
        sd = tr.rolling(rolling).std()
        return ((tr - m) / sd).dropna()

    def test_rank_xsect(self):
        """Test cross-sectional rank using RankXSect tform"""

        # Run rank tform
        ranks = aspen.tform.library.rank.RankXSect(ascending=True, pct=False).apply(self.zsc)
        pct = aspen.tform.library.rank.RankXSect(pct=True).apply(self.zsc)

        # Assertion statements
        pd.testing.assert_series_equal(
            self.zsc.iloc[4].rank(ascending=True, pct=False), ranks.iloc[4]
        )
        pd.testing.assert_series_equal(
            self.zsc.iloc[5].rank(ascending=False, pct=True), pct.iloc[5]
        )

    def test_qcut_zsect(self):
        """Test cross-sectional bins using QCutXSect tform"""

        # Run qcut tform
        qcut_3 = aspen.tform.library.rank.QCutXSect(bins=3).apply(self.zsc)
        qcut_5 = aspen.tform.library.rank.QCutXSect(bins=5).apply(self.zsc)

        # Assertion statements
        self.assertEqual(qcut_3.max().max(), 3)
        self.assertEqual(qcut_3.min().max(), 1)
        self.assertEqual(qcut_5.max().max(), 5)
        self.assertEqual(qcut_5.min().max(), 1)

        pd.testing.assert_series_equal(
            pd.qcut(self.zsc.iloc[4], q=3, labels=[1, 2, 3]), qcut_3.iloc[4],
        )
        pd.testing.assert_series_equal(
            pd.qcut(self.zsc.iloc[5], q=5, labels=[1, 2, 3, 4, 5]), qcut_5.iloc[5],
        )

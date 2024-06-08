"""Unit tests for Pipeline"""

import unittest

import numpy as np
import pandas as pd

from tform import TForm, Pipeline
import tform.generic


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
            TForm("sum", mode=tform.generic.Mode.PASS, lib="numpy"),
            TForm("mean", mode=tform.generic.Mode.PASS, lib="numpy")
        )
        self.assertEqual(pl.apply(self.data), 8.0)

    def test_diagnostics(self):
        """Test diagnostics saved correctly"""
        pl = Pipeline(TForm("sum"), TForm("mean"), save=True)
        res = pl.apply(self.data)

        self.assertEqual(len(pl.diagnostics), 3)

        pd.testing.assert_frame_equal(pl.diagnostics[0][1], self.data)
        self.assertEqual(pl.diagnostics[-1][1], res)

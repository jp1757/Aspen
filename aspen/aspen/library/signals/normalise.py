"""
INormalise implementations
"""

import pandas as pd

import aspen.library.tform.rank
from aspen.signals.core import INormalise
from aspen.tform.core import ITForm


class XSWeights(INormalise):
    """Weight proportionally to an input score so absolute values sum to 1.0"""

    def norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise signal data
        :param data: (pd.DataFrame) signal data to normalise
        :return: (pd.DataFrame) normalised signal data
        """
        return data.div(data.abs().sum(axis=1), axis=0).sort_index()

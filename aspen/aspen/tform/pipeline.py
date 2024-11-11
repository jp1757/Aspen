"""
Sets out logic for combining multiple transformations in succession
aka function composition
"""

import pandas as pd

from aspen.tform.core import ITForm


class Pipeline(ITForm):
    """
    A class that applies multiple transformations to an input dataframe
    in succession akin to function composition
    """

    def __init__(self, *tform: ITForm, save: bool = False) -> None:
        """
        Init Pipeline object
        :param tform: (ITForm) >= 1 ITForm object to run in succession
        :param save: (bool, optional) save state after each transformation
        """
        self.tforms: ITForm = tform
        self.save: bool = save

        self.diagnostics = []
        self.__apply = self._diagnostics if save else self._apply

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run pipeline on input data

        :param data: (pandas.DataFrame) input data to apply transformation to
        :param *other: (pandas.DataFrame) other data frames to use in transformation
        :return: (pandas.DataFrame) transformed data
        """
        return self.__apply(data)

    def _apply(self, data: pd.DataFrame) -> pd.DataFrame:
        for tf in self.tforms:
            data = tf.apply(data)

        return data

    def _diagnostics(self, data: pd.DataFrame) -> pd.DataFrame:
        self.diagnostics = []
        self.diagnostics.append((None, data))

        for tf in self.tforms:
            data = tf.apply(data)
            self.diagnostics.append((tf, data))

        return data

"""
Provides a definition for a Signal object
"""
import pandas as pd

from signals import ISignal


class Signal(ISignal):
    """
    Generic object for building signals from a group of transformations
    """

    def __init__(self, pre, post):
        pass

    def calculate(self, **data: pd.DataFrame):
        pass

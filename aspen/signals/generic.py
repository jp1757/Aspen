"""
Provides a definition for a Signal object
"""
import pandas as pd
from typing import List, Dict

from signals import ISignal, Leaf
from tform import ITForm


class Signal(ISignal):
    """
    Generic object for building signals from a group of transformations
    """

    def __init__(
            self, *leaves: Leaf, data: Dict[str, pd.DataFrame], post: List[Leaf] = None,
    ) -> None:
        self.leaves = leaves
        self.data = data
        self.post = [] if post is None else post

    def calculate(self) -> pd.DataFrame:

        signal = None
        for leaf in self.leaves:
            signal = leaf.build(self.data)

        return signal
    
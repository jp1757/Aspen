"""
Provides a definition for a Signal object
"""
from typing import Dict

import pandas as pd

from aspen.signals import ISignal, ILeaf


class Signal(ISignal):
    """
    Generic object for building signals from a group of transformations
    """

    def __init__(self, *leaves: ILeaf, data: Dict[str, pd.DataFrame], name: str) -> None:
        self.leaves = leaves
        self.data = data
        self.__name = name

    @property
    def name(self) -> str:
        """Unique signal id"""
        return self.__name

    def calculate(self) -> pd.DataFrame:
        signal = None
        for leaf in self.leaves:
            # Apply transformation
            signal = leaf.build(signal, **self.data)

            # Add transformed to data heap using leaf's unique key
            self.data[leaf.key] = signal

        return signal

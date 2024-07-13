"""
Provides a definition for a Signal object
"""
from typing import Dict

import pandas as pd

from aspen.signals.core import ISignal, ISignals
from aspen.signals.leaf import ILeaf


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


class Signals(ISignals):
    """
    Stores multiple signal objects but does *not* combine them.
    Access them by passing the signal name to build function
    """

    def __init__(self, *signals: ISignal):

        self._signals = {s.name: s for s in signals}
        if len(signals) > len(self._signals):
            raise ValueError(
                f"Duplicate signal name suspected: {[s.name for s in signals]}"
            )

    def _combine(self) -> pd.DataFrame:
        """Override this function with specific signal combination logic"""
        raise ValueError(
            "This implementation does combine signals. Please set the name "
            "parameter to retrieve specific signal data"
        )

    def build(self, name: str = None) -> pd.DataFrame:
        """
        Serve up signal data by setting the 'name' parameter

        :param name: (str, optional) name of signal to return
        :return: pd.DataFrame of signal data indexed by date with columns set
            to asset ids
        """

        if name is None:
            return self._combine()
        else:
            return self._signals[name].calculate()

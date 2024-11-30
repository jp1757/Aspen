"""
Provides a definition for a SignalHeap object
"""

import functools
from enum import Enum
from typing import Dict, List

import pandas as pd

from aspen.signals.core import ISignal, ISignals, INormalise
from aspen.signals.leaf import ILeaf
from aspen.tform.core import ITForm


class SignalType(Enum):
    DIRECTIONAL = 1
    REVERSION = 2


class Signal(ISignal):
    """
    Generic object for building signals from a group of transformations,
    calculate from a single data object
    """

    def __init__(self, *leaves: ILeaf, data: pd.DataFrame, name: str) -> None:
        self.leaves = leaves
        self.data = data
        self.__name = name

    @property
    def name(self) -> str:
        """Unique signal id"""
        return self.__name

    def calculate(self) -> pd.DataFrame:
        signal = self.data
        for leaf in self.leaves:
            # Apply transformation
            signal = leaf.build(signal)

        return signal


class SignalHeap(ISignal):
    """
    Generic object for building signals from a group of transformations,
    calculated from multiple data inputs
    """

    def __init__(
        self, *leaves: ILeaf, data: Dict[str, pd.DataFrame], name: str
    ) -> None:
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


class SignalDF(ISignal):
    """
    Basic signal object that holds signal data already derived & stored in a dataframe
    """

    def __init__(self, name: str, data: pd.DataFrame) -> None:
        self._name = name
        self._data = data

    @property
    def name(self) -> str:
        return self._name

    def calculate(self) -> pd.DataFrame:
        return self._data


class Signals(ISignals):
    """
    Stores multiple signal objects but does *not* combine them.  Please inherit &
    override _combine function with logic for combining signals.
    This class just provides access to the individual signals by passing the name to
    the build function
    """

    def __init__(
        self,
        *signals: ISignal,
        name: str,
        direction: SignalType,
        normalise: INormalise = None,
    ) -> None:
        """
        Init generic Signals object used for combining ISignal objects
        :param signals: (ISignal) >= 1 ISignal object to store
        :param name: (str) name of signals object
        :param normalise: (INormalise, optional) option for normalising signal data
            when build is called as the final step.
        """

        self._signals = {s.name: s for s in signals}
        if len(signals) > len(self._signals):
            raise ValueError(
                f"Duplicate signal name suspected: {[s.name for s in signals]}"
            )
        self._name = name
        self.normalise = normalise

        # Check direction
        if not isinstance(direction, SignalType):
            raise ValueError(
                "Invalid direction please use type .signals.generic.SignalType"
            )
        self.direction = direction

    @property
    def name(self) -> str:
        return self._name

    @property
    def signals(self) -> List[ISignal]:
        """Return list of individual signals"""
        return list(self._signals.values())

    def _combine(self) -> pd.DataFrame:
        """Override this function with specific signal combination logic"""
        raise ValueError(
            "This implementation does combine signals. Please set the name "
            "parameter to retrieve specific signal data"
        )

    @functools.lru_cache(maxsize=None)
    def build(self, name: str = None) -> pd.DataFrame:
        """
        Serve up signal data by setting the 'name' parameter

        :param name: (str, optional) name of signal to return
        :return: pd.DataFrame of signal data indexed by date with columns set
            to asset ids
        """

        _s = (
            list(self._signals.values())[0].calculate()
            if len(self._signals) == 1
            else (self._combine() if name is None else self._signals[name].calculate())
        )
        if self.normalise is not None:
            _s = self.normalise.norm(_s)

        # Convert to signal direction
        if self.direction == SignalType.REVERSION:
            _s = _s.mul(-1)

        return _s


class SignalsDF(ISignals):
    """
    Store & return a pre-calculated signals dataframe
    """

    def __init__(self, name: str, data: pd.DataFrame) -> None:
        """
        Init object and store data
        :param name: (str) name of signal data
        :param data: (pd.DataFrame) signal data
        """
        self._name = name
        self.data = data

    @property
    def name(self) -> str:
        return self._name

    @property
    def signals(self) -> List[ISignal]:
        """Return list of individual signals"""
        return [SignalDF(name=self.name, data=self.data)]

    def build(self, name: str = None) -> pd.DataFrame:
        """
        Serve up signal data by setting the 'name' parameter

        :param name: (str, optional) name of signal to return
        :return: pd.DataFrame of signal data indexed by date with columns set
            to asset ids
        """

        return self.data


class Normalise(INormalise):
    """
    Generic implementation of INormalise that accepts ITForm objects.
    Basically acting as a ITForm->INormalise adapter used for post
    signal processing before backtests.
    """

    def __init__(self, tform: ITForm):
        self.tform = tform

    def norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise signal data
        :param data: (pd.DataFrame) signal data to normalise
        :return: (pd.DataFrame) normalised signal data
        """
        return self.tform.apply(data)

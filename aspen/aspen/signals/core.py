"""
Sets out the core SignalHeap interface
"""

import abc

import pandas as pd


class ISignal(metaclass=abc.ABCMeta):
    """
    Core Individual SignalHeap Interface
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique signal id"""
        pass

    @abc.abstractmethod
    def calculate(self) -> pd.DataFrame:
        """
        Calculate a signal object & return data
        :return: pd.DataFrame
        """
        pass


class ISignals(metaclass=abc.ABCMeta):
    """
    Sets out a structure for normalising & then combining
    multiple individual signal (ISignal) objects.
    """

    @abc.abstractmethod
    def build(self, name: str = None) -> pd.DataFrame:
        """
        Serve up signal data by either combining multiple signals or
        returning a specific signal by setting the 'name' parameter

        :param name: (str, optional) name of signal to return
        :return: pd.DataFrame of signal data indexed by date with columns set
            to asset ids
        """
        pass


class INormalise(metaclass=abc.ABCMeta):
    """
    Sets out structure of signal normalisation objects
    """

    @abc.abstractmethod
    def norm(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise signal data
        :param data: (pd.DataFrame) signal data to normalise
        :return: (pd.DataFrame) normalised signal data
        """

"""
Sets out the core Signal interface
"""

import abc

import pandas as pd


class ISignal(metaclass=abc.ABCMeta):
    """
    Core Individual Signal Interface
    """

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
    def combine(self) -> pd.DataFrame:
        """
        Combine multiple signals
        :return: pd.DataFrame
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

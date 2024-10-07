"""
Contains objects for linking ITForms to SignalsDummy
"""
import abc
from typing import List

import pandas as pd

from aspen.tform.core import ITForm


class ILeaf(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def key(self) -> str:
        """Leaf name used to add transformed data to heap"""
        pass

    def build(self, data: pd.DataFrame, **heap: pd.DataFrame) -> pd.DataFrame:
        """
        Build each transformation passing either just the input dataframe or
        mapping multiple dataframes from the additional heap (dict)

        :param data: (pd.DataFrame) latest transformed data
        :param heap: (pandas.DataFrame) dictionary of all data

        :return: (pandas.DataFrame) transformed data
        """
        pass


class Leaf(ILeaf):
    """
    Acts as a SignalHeap compatible wrapper for an ITForm object mapping
    data from the input heap
    """

    def __init__(self, tform: ITForm, key: str, mappings: List[str]) -> None:
        """
        Init Leaf instance

        :param tform: (ITForm) instance containing transformation logic to pass data to
        :param key: (str) Leaf's unique name used to add its output to heap
        :param mappings: (list(str)) list of keys that map to dataframes in heap.  Data will
            be extracted from heap & passed to ITForm.apply in same order
            i.e. ITForm.apply(*[heap.get(x) for x in mappings if x in heap]).
            'data' cannot be one of the mappings as it will conflict with 'data' param in
            self.build
        """
        self.tform = tform
        self.__key = key
        self.mappings = mappings

        # Check mappings doesn't contain the key 'data'
        if 'data' in mappings:
            raise ValueError(f"Do not use the key 'data' in mapppings: {mappings}")

    @property
    def key(self) -> str:
        """Leaf name used to add transformed data to heap"""
        return self.__key

    def build(self, data: pd.DataFrame, **heap: pd.DataFrame) -> pd.DataFrame:
        """
        Build each transformation passing either just the input dataframe or
        mapping multiple dataframes from the additional heap (dict)

        :param data: (pd.DataFrame) latest transformed data
        :param heap: (pandas.DataFrame) dictionary of all data

        :return: (pandas.DataFrame) transformed data
        """

        mapped = [heap.get(x) for x in self.mappings if x in heap]

        # Check all mappings found in heap
        if len(mapped) != len(self.mappings):
            missing = set(self.mappings) - set(heap.keys())
            raise ValueError(f"Mappings not found in heap: {missing}")

        # Apply transformation & return
        return self.tform.apply(*mapped)


class LeafSeries(ILeaf):
    """
    Acts as a SignalHeap compatible wrapper for an ITForm object passing transformed
    data from previous transformation
    """

    def __init__(self, tform: ITForm, key: str) -> None:
        """
        Init LeafSeries instance

        :param tform: (ITForm) instance containing transformation logic to pass data to
        :param key: (str) Leaf's unique name used to add its output to heap
        """
        self.tform = tform
        self.__key = key

    @property
    def key(self) -> str:
        """Leaf name used to add transformed data to heap"""
        return self.__key

    def build(self, data: pd.DataFrame, **heap: pd.DataFrame) -> pd.DataFrame:
        """
        Build each transformation passing either just the input dataframe or
        mapping multiple dataframes from the additional heap (dict)

        :param data: (pd.DataFrame) latest transformed data
        :param heap: (pandas.DataFrame) dictionary of all data

        :return: (pandas.DataFrame) transformed data
        """

        # Apply transformation & return
        return self.tform.apply(data)

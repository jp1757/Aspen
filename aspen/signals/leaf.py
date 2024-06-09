"""
Contains objects for linking TForms to Signals
"""
from typing import Dict, List

import pandas as pd

from tform.core import ITForm


class Leaf:
    """
    Acts as a wrapper to a TForm mapping data from the input heap
    """

    def __init__(
            self, tform: ITForm, key: str, mappings: List[str], final: bool = False,
    ) -> None:
        self.tform = tform
        self.key = key
        self.mappings = mappings
        self.__final = final

    @property
    def final(self) -> bool:
        return self.__final

    def build(self, heap: Dict[str: pd.DataFrame]) -> pd.DataFrame:
        """
        Build each transformation mapping data from the heap

        :param heap: (pandas.DataFrame) input data to apply transformation to
        :return: (pandas.DataFrame) transformed data
        """

        # Get data from heap
        _data = [heap.get(x) for x in self.mappings if x in self.heap]

        # Apply transformation
        tfdata = self.tform.apply(*_data)

        # Add to the heap
        self.heap[self.key] = tfdata

        return tfdata

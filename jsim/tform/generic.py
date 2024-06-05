"""
Generic TForm container pointing to any lib
"""
import pandas as pd
import importlib

import tform.core


class Generic(tform.core.ITForm):

    def __init__(self, type: str, func: str, lib: str = None, **kwargs):
        """
        Init generic instance

        :param type: (str) following two modes accepted:
            - 'call' - input data object calls function i.e. data.sum()
            - 'args' - pass data object as input to target function  i.e. numpy.sum(data)
        :param func: (str) name of target function
        :param lib: (str, optional) name of target lib
        :param kwargs: (dict, optional) additional keyword args to pass to target function
        """

        # Check Params
        self.__mode = self.TYPES.get(type.lower(), None)
        if self.__mode is None:
            raise ValueError(
                f"Type [{type}] not valid.  Accepted values: {self.TYPES.keys()}"
            )

        # Store instance paras
        self.type = type
        self.func = func
        self.kwargs = kwargs

        # Load library
        self.lib = importlib.import_module(lib) if lib is not None else None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation using target function passed to __init__ to input data

        :param data: (pd.DataFrame) data to transform
        :return: (pd.DataFrame) transformed data
        """
        return self.__mode(data)

    def call(self, data: pd.DataFrame) -> pd.DataFrame:
        """Call target function on data"""
        return getattr(data, self.func)(**self.kwargs)

    def args(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pass data as an argument to target function"""
        return getattr(self.lib, self.func)(data, **self.kwargs)

    # Modes
    TYPES = {
        "call": call,
        "pass": args
    }

"""
Generic TForm container pointing to any lib
"""
import enum
import importlib

import pandas as pd

import tform.core


class Mode(enum.Enum):
    CALL = 1
    PASS = 2


class Generic(tform.core.ITForm):

    def __init__(self, mode: Mode, func: str, *, lib: str = None, **kwargs) -> None:
        """
        Init generic instance

        :param mode: (Mode) following two modes accepted:
            - 'CALL' - input data object calls function i.e. data.sum()
            - 'PASS' - pass data object as input to target function  i.e. numpy.sum(data)
        :param func: (str) name of target function
        :param lib: (str, optional) name of target lib
        :param kwargs: (dict, optional) additional keyword args to pass to target function
        """

        # Check Params
        self.__mode = self.MODES.get(mode.name, None)
        if self.__mode is None:
            raise ValueError(
                f"Type [{mode.name}] not valid.  Accepted values: {self.MODES.keys()}"
            )

        # Store instance paras
        self.type = mode
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
        return self.__mode(self, data)

    def call(self, data: pd.DataFrame) -> pd.DataFrame:
        """Call target function on data"""
        return getattr(data, self.func)(**self.kwargs)

    def args(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pass data as an argument to target function"""
        return getattr(self.lib, self.func)(data, **self.kwargs)

    # Modes
    MODES = {
        "CALL": call,
        "PASS": args
    }


class Merge(tform.core.IMerge):

    def __init__(self, mode: Mode, func: str, *, lib: str = None, **kwargs) -> None:
        """
        Init Merge instance

        :param mode: (Mode) following two modes accepted:
            - 'CALL' - 1st data object in args calls function & rest are passed
                       as positional args i.e. data[0].func(*data[1:])
            - 'PASS' - pass data objects as inputs to target function
                       i.e. func(*data)
        :param func: (str) name of target function
        :param lib: (str, optional) name of target lib
        :param kwargs: (dict, optional) additional keyword args to pass to target function
        """

        # Check Params
        self.__mode = self.MODES.get(mode.name, None)
        if self.__mode is None:
            raise ValueError(
                f"Type [{mode.name}] not valid.  Accepted values: {self.MODES.keys()}"
            )

        # Store instance paras
        self.type = mode
        self.func = func
        self.kwargs = kwargs

        # Load library
        self.lib = importlib.import_module(lib) if lib is not None else None

    def merge(self, *data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge multiple dataframes

        :param data: (List[pd.DataFrame]) input data args to merge
        :return: (pd.DataFrame) merged dataframe
        """
        return self.__mode(self, *data)

    def call(self, *data: pd.DataFrame) -> pd.DataFrame:
        """Call target function on data"""
        return getattr(data[0], self.func)(*data[1:], **self.kwargs)

    def args(self, *data: pd.DataFrame) -> pd.DataFrame:
        """Pass data as an argument to target function"""
        return getattr(self.lib, self.func)(*data, **self.kwargs)

    # Modes
    MODES = {
        "CALL": call,
        "PASS": args
    }

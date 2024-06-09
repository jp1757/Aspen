"""
Generic containers that can use any library function to apply a transformation
to either a single ('TForm') or multiple ('Merge') input dataframes
"""
import abc
import enum
import importlib

import pandas as pd

import tform.core


class Mode(enum.Enum):
    CALL = 1
    PASS = 2


class Generic(metaclass=abc.ABCMeta):

    def __init__(
            self, func: str, *, mode: Mode = Mode.CALL, lib: str = None, **kwargs
    ) -> None:
        """
        Generic constructor - see child instances for mode implementation

        :param func: (str) name of target function
        :param mode: (Mode, optional) determines if the data obj(s) call the target
            function or are passed to the target function. Default is set to Mode.CALL
        :param lib: (str, optional) name of target lib
        :param kwargs: (dict, optional) additional keyword args to pass to target function
        """

        # Check Params
        if not isinstance(mode, tform.generic.Mode):
            raise ValueError(f"Mode [{mode}] must be of type [{Mode}]")

        # Store instance paras
        self.mode = mode
        self.func = func
        self.kwargs = kwargs

        # Load library
        self.lib = importlib.import_module(lib) if lib is not None else None


class TForm(tform.core.ITForm, Generic):
    """
    TForm instance for applying a transformation ot a single dataframe

    The following illustrates how the constructor mode param performs:

    :param mode: (Mode) following two modes accepted:
        - 'CALL' - input data object calls function i.e. data.sum()
        - 'PASS' - pass data object as input to target function  i.e. numpy.sum(data)
    """

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: (pandas.DataFrame) input data to apply transformation to
        :param *other: (pandas.DataFrame) other data frames to use in transformation

        :return: (pandas.DataFrame) transformed data
        """

        if self.mode == Mode.CALL:
            """Call target function on data"""
            return getattr(data, self.func)(**self.kwargs)

        elif self.mode == Mode.PASS:
            """Pass data as an argument to target function"""
            return getattr(self.lib, self.func)(data, **self.kwargs)


class Merge(tform.core.ITForm, Generic):
    """
    Merge instance for applying a transformation to multiple dataframes.

    The following illustrates how the constructor mode param performs:

    :param mode: (Mode) following two modes accepted:
        - 'CALL' - 1st data object in args calls function & rest are passed
                   as positional args i.e. data[0].func(*data[1:])
        - 'PASS' - pass data objects as inputs to target function
                   i.e. func(*data)
    """

    def apply(self, data: pd.DataFrame, *other: pd.DataFrame) -> pd.DataFrame:
        """
        Merge multiple dataframes via transformation func passed to __init__

        :param data: (pandas.DataFrame) input data to apply transformation to
        :param *other: (pandas.DataFrame) other data frames to use in transformation

        :return: (pandas.DataFrame) transformed data
        """
        if self.mode == Mode.CALL:
            """Call target function on data"""
            return getattr(data, self.func)(*other, **self.kwargs)

        elif self.mode == Mode.PASS:
            """Pass data as an argument to target function"""
            return getattr(self.lib, self.func)(data, *other, **self.kwargs)

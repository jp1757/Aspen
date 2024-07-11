"""
A set of statistics for validating the efficacy of signals
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd

from aspen.tform.library.align import Reindex


def __align_returns(
        *,
        tr: Union[pd.Series, pd.DataFrame],
        signal: Union[pd.Series, pd.DataFrame],
        lag: int
) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
    """
    Calculate & align returns data with signal data to calculate correlations

    :param tr: (pd.Series) total return price series of asset to calculate forward
        returns from i.e [1.0, 1.01, 0.98...]
    :param signal: (pd.Series) signal series to test predictive capabilities of
    :param lag: (int) number of forward return periods to compare against signal.
        i.e. if monthly tr data, 1 means 1 month forward return, 3 means 1 quarter
        forward return.
    :return: either two pd.Series or two pd.DataFrame of signal & shifted returns data
    """
    returns = Reindex(signal.index).apply(tr).pct_change(periods=lag)
    shifted = returns.shift(-lag).dropna(how="all")
    signal = Reindex(shifted.index).apply(signal)

    return signal, shifted


def ic(
        tr: pd.Series, *, signal: pd.Series, lag: int, rolling: int
) -> Tuple[float, pd.Series]:
    """
    Calculate the information coefficient (IC) and rolling IC of a signal, looking at the
    correlation between signal values and forward returns.

    :param tr: (pd.Series) total return price series of asset to calculate forward
        returns from i.e [1.0, 1.01, 0.98...]
    :param signal: (pd.Series) signal series to test predictive capabilities of
    :param lag: (int) number of forward return periods to compare against signal.
        i.e. if monthly tr data, 1 means 1 month forward return, 3 means 1 quarter
        forward return.
    :param rolling: (int) rolling IC period value
    :return: Tuple(float, pd.Series) correlation over entire period of signal and
        forward returns, plus series of rolling correlation values
    """
    signal, shifted = __align_returns(tr=tr, signal=signal, lag=lag)

    corr = signal.corr(shifted)
    rolling = signal.rolling(rolling).corr(shifted).dropna()

    return corr, rolling


def ic_xsect(
        tr: pd.DataFrame, *, signal: pd.DataFrame, lag: int, rank: str = None,
) -> pd.Series:
    """
    Calculate the information coefficient (IC) and rolling IC of a signal, looking
    at the correlation between signal values and forward returns.

    :param tr: (pd.DataFrame) total return prices of assets to calculate forward
        returns from i.e [1.0, 1.01, 0.98...]. Column names should be set to assets
        & match the signal dataframe.
    :param signal: (pd.DataFrame) signal data to test predictive capabilities of.
        Column names should be set to assets & match the tr dataframe.
    :param lag: (int) number of forward return periods to compare against signal.
        i.e. if monthly tr data, 1 means 1 month forward return, 3 means 1 quarter
        forward return.
    :param rank: (str, optional) whether to rank signal values before calculating
        correlations. "basic": uses pd.rank(), "pct": uses pd.rank(pct=True)
    :return: (pd.Series) cross-sectional correlation between signal & forward
        returns on each date
    """
    signal, shifted = __align_returns(tr=tr, signal=signal, lag=lag)

    if rank is not None:
        rank = rank.lower()
        if rank not in ["basic", "pct"]:
            raise ValueError("Invalid rank type check docstring; use 'basic' or 'pct'")
        signal = signal.rank(axis=1, pct=(rank == "pct"))

    corr = signal.corrwith(shifted, axis=1)

    return corr


def tstat(scores: pd.Series) -> float:
    """
    Calculate the tstat of a series of scores
    :param scores: (pd.Series) series of scores
    :return: (float) T-Stat value
    """
    return scores.mean() / scores.std() * np.sqrt(len(scores))


def quintiles():
    pass

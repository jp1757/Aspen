"""
A set of statistics for validating the efficacy of signals
"""

from typing import Tuple

import numpy as np
import pandas as pd

from aspen.tform.library.align import Reindex


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

    returns = Reindex(signal.index).apply(tr).pct_change(periods=lag)
    shifted = returns.shift(-lag).dropna()
    signal = Reindex(shifted.index).apply(signal)

    corr = signal.corr(shifted)
    rolling = signal.rolling(rolling).corr(shifted).dropna()

    return corr, rolling


def tstat(scores: pd.Series) -> float:
    """
    Calculate the tstat of a series of scores
    :param scores: (pd.Series) series of scores
    :return: (float) T-Stat value
    """
    return scores.mean() / scores.std * np.sqrt(len(scores))

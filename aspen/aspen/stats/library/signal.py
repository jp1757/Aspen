"""
A set of statistics for validating the efficacy of signals
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import sklearn.linear_model

from aspen.tform.library.align import Reindex, Align


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
        correlations. "basic": uses pd.rank(), "pct_rank": uses pd.rank(pct_rank=True)
    :return: (pd.Series) cross-sectional correlation between signal & forward
        returns on each date
    """
    signal, shifted = __align_returns(tr=tr, signal=signal, lag=lag)

    if rank is not None:
        rank = rank.lower()
        if rank not in ["basic", "pct_rank"]:
            raise ValueError("Invalid rank type check docstring; use 'basic' or 'pct_rank'")
        signal = signal.rank(axis=1, pct=(rank == "pct_rank"))

    corr = signal.corrwith(shifted, axis=1)

    return corr


def tstat(scores: pd.Series) -> float:
    """
    Calculate the tstat of a series of scores
    :param scores: (pd.Series) series of scores
    :return: (float) T-Stat value
    """
    return scores.mean() / scores.std() * np.sqrt(len(scores))


def success_rate(scores: pd.Series) -> float:
    """
    Calculate the success rate of a series of ICs.
    number of months IC > 0 / total number of months

    :param scores: (pd.Series) series of scores
    :return: (float) success rate
    """
    return len(scores[scores > 0]) / len(scores)


def pure_factor(
        factor: pd.DataFrame, *others: pd.DataFrame, tr: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate a pure factor return by stripping out exposures to traditional risk or
    other relevant factors.  We do this by performing a multivariate cross-sectional
    regression on each date.  The regression coefficient of the selected factor is
    used to proxy the pure factor return.  In essence this is the equivelent return
    when taking a 1 STD exposure to the factor whilst stripping out exposure to the others

    :param factor: (pd.DataFrame) factor to calculate pure returns for. Column names
        should be set to assets, with dates in the index.
    :param others: (pd.DataFrame) factors to strip out. Column names should be set to
        assets, with dates in the index.
    :param tr: (pd.DataFrame) total return prices of assets to calculate forward
        returns from i.e [1.0, 1.01, 0.98...]. Column names should be set to assets
        & match the signal dataframe.

    :return: (Tuple[pd.Series, pd.Series]) a series of pure factor returns, a series
        of pure factor total return prices
    """

    # Align factors with returns
    align = Align(*[x.index for x in others] + [factor.index] + [tr.index])
    ret_1M = align.apply(tr).pct_change().shift(-1).fillna(0)
    reindex = Reindex(ret_1M.index)
    factors = [reindex.apply(x) for x in [factor] + list(others)]

    # Combine factors
    df = pd.concat([r.stack() for r in factors], axis=1)

    # Using sklearn
    # Loop through dates calculating cross-sectional regression of factors vs
    # forward returns
    frets = {
        _dt: sklearn.linear_model.LinearRegression().fit(df.loc[_dt], _rets).coef_[0]
        for _dt, _rets in ret_1M.iterrows()
    }

    # Shift returns back forward
    fret = pd.Series(frets).shift(1)
    fret.iloc[0] = 0
    ftr = (1 + fret).cumprod()
    fret.iloc[0] = np.NaN

    return fret, ftr

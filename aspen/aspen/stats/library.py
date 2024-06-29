"""
Library of portfolio statistics calculations wrapped as
TForm objects
"""
import warnings
from typing import Union

import numpy as np
import pandas as pd


def __check(tr: pd.Series, func: str):
    if tr.isna().any():
        warnings.warn(
            f"NaN values found in total return prices from func: [{func}]. "
            f"Filling forward last values"
        )
        tr = tr.ffill()

    return tr


def cagr(*, tr: pd.Series, periods: int, rolling: int = None) -> Union[float, pd.Series]:
    """
    Calculate the compounded average growth rate for an input series

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :param periods: (int) periods per year i.e. 12 for monthly, 252 for daily etc.
    :param rolling: (int, optional) use to calculate rolling CAGRs
    :return: either scalar or pd.Series when calculating rolling CAGRs
    """

    tr = __check(tr, "cagr")

    length = (len(tr) if rolling is None else rolling) - 1
    returns = tr.pct_change(length).dropna().squeeze() + 1

    return np.power(returns, (periods / length)) - 1


def volatility(*, tr: pd.Series, periods: int):
    tr = __check(tr, "volatility")
    return np.std(tr.pct_change()) * np.sqrt(periods)


def sharpe(*, tr: pd.Series, periods: int, rfr: float = 0.0):
    tr = __check(tr, "sharpe")

    # Calculate risk-free rate
    if rfr > 0:
        rfr = deannualize(rate=rfr, periods=periods)

    # Calculate returns
    returns = tr.pct_change()
    # Calculate excess returns vs rfr
    excess = returns - rfr
    excess.iloc[0] = 0
    # Calculate compounded tr prices from excess
    compound = (1 + excess).cumprod()
    # Get CAGR
    _cagr = cagr(tr=compound, periods=periods)
    # Get vol
    vol = volatility(tr=compound, periods=periods)

    return _cagr / vol


def deannualize(*, rate: float, periods: int):
    return np.power(rate + 1, 1 / periods) - 1

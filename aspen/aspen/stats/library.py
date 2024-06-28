"""
Library of portfolio statistics calculations wrapped as
TForm objects
"""
import warnings

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


def cagr(*, tr: pd.Series, periods: int):
    tr = __check(tr, "cagr")
    return np.power(tr.iloc[-1] / tr.iloc[0], (periods / (len(tr) - 1))) - 1


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

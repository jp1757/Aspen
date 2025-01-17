"""
Statistics calculations on time series data
"""
import warnings
from typing import Union

import numpy as np
import pandas as pd

import aspen.stats.library.scalar
from aspen.library.tform.align import Align


def __check(tr: pd.Series, func: str):
    if tr.isna().any().any():
        warnings.warn(
            f"NaN values found in total return prices from func: [{func}]. "
            f"Filling forward last values"
        )
        tr = tr.ffill()

    return tr


def tr(tr: pd.Series) -> pd.Series:
    """
    Calculate the compound total return

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :return: pd.Series of compounded total returns
    """
    _tr = __check(tr, "tr")
    ret = _tr.pct_change()
    ret.iloc[0] = 0

    return (1 + ret).cumprod() - 1


def year_return(tr: pd.Series, *, years: int, periods: int) -> float:
    """
    Calculate a return for the last x years

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :param years: (int) period to calculate return over
    :param periods: (int) periods per year i.e. 12 for monthly, 252 for daily etc.
    :return: (float) return over last x years
    """
    tr = __check(tr, "ret")

    start_idx = years * periods + 1
    if start_idx > len(tr):
        warnings.warn(
            f"stats.ret: Doesn't have [{years}] years worth of data"
        )
        start_idx = len(tr)

    return tr.iloc[-1] / tr[-start_idx] - 1


def cagr(tr: pd.Series, *, periods: int, rolling: int = None) -> Union[float, pd.Series]:
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


def vol(tr: pd.Series, *, periods: int, rolling: int = None) -> Union[float, pd.Series]:
    """
    Calculate annualised volatility for an input series

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :param periods: (int) periods per year i.e. 12 for monthly, 252 for daily etc.
    :param rolling: (int, optional) use to calculate rolling volatility
    :return: either scalar or pd.Series when calculating rolling volatility
    """
    tr = __check(tr, "vol")
    length = (len(tr) if rolling is None else rolling) - 1
    stds = tr.pct_change().rolling(length).std(ddof=1)

    return (stds * np.sqrt(periods)).dropna().squeeze()


def sharpe(
        tr: pd.Series, *, periods: int, rfr: float = 0.0, rolling: int = None,
) -> Union[float, pd.Series]:
    """
    Calculate Sharpe Ratio

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :param periods: (int) periods per year i.e. 12 for monthly, 252 for daily etc.
    :param rfr: (float, optional) risk-free rate to subtract from returns
    :param rolling: (int, optional) use to calculate rolling sharpe
    :return: either scalar or pd.Series when calculating rolling sharpe
    :return:
    """

    tr = __check(tr, "sharpe")

    # Calculate risk-free rate
    if rfr > 0:
        rfr = aspen.stats.library.scalar.deannualize(rate=rfr, periods=periods)

    # Calculate returns or excess returns if risk-free-rate passed
    if rfr > 0:
        xs = excess(tr=tr, other=rfr)
        xs.iloc[0] = 0
        tr = (1 + xs).cumprod()

    # Get CAGR
    _cagr = cagr(tr=tr, periods=periods, rolling=rolling)

    # Get vol
    _vol = vol(tr=tr, periods=periods, rolling=rolling)

    return _cagr / _vol


def excess(tr: pd.Series, *, other: Union[float, pd.Series]) -> pd.Series:
    """
    Calculate the excess returns above something else.  Can be a fixed rate
    or another series of returns

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :param other: (float or pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :return: pd.Series of excess returns
    """
    __check(tr, "excess")

    if isinstance(other, pd.Series):
        __check(tr, "excess[other]")
        _other = Align(tr.index).apply(other)
        other = _other.loc[tr.index.min(): min(tr.index.max(), other.index.max())]
        other = other.pct_change()

    xs = tr.pct_change() - other
    xs.iloc[0] = 0
    xs.dropna(inplace=True)
    xs.iloc[0] = np.NaN
    return xs


def drawdown(tr: pd.Series, *, periods: int = None, rfr: float = 0.0) -> pd.Series:
    """
    Calculate underwater curve

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :param periods: (int, optional) periods per year i.e. 12 for monthly, 252 for daily etc.
        Must be set if passing rfr, so that rate can be de-annualized
    :param rfr: (float, optional) risk-free rate to subtract from returns
    :return: pd.Series of drawdown values
    """
    tr = __check(tr, "drawdown")

    # Calculate risk-free rate
    if rfr > 0:
        if periods is None:
            raise ValueError("Please pass a value for periods if rfr set")
        rfr = aspen.stats.library.scalar.deannualize(rate=rfr, periods=periods)

    # Subtract risk-free-rate
    if rfr > 0:
        xs = excess(tr=tr, other=rfr)
        xs.iloc[0] = 0
        tr = (1 + xs).cumprod()

    # Calculate drawdown
    return tr / tr.expanding().max() - 1


def turnover(
        weights: pd.DataFrame, *, periods: int, drifted: pd.DataFrame = None
) -> float:
    """
    Calculate the total two-sided turnover average per year.  Buys + sells.

    :param weights: (pd.DataFrame) portfolio weights indexed by date, with assets as columns
    :param periods: (int, optional) periods per year i.e. 12 for monthly, 252 for daily etc.
    :param drifted: (pd.DataFrame, optional) calculate turnover using drifted weights taking
        into consideration the drift between rebalance dates.
    :return: float average annual turnover
    """

    if drifted is not None:
        dates_diff = set(weights.index) - set(drifted.index)
        if len(dates_diff) > 0:
            raise ValueError(
                f"Dates found in weights dataframe not present in drifted "
                f"weights: {dates_diff}"
            )
        diff = drifted.diff().loc[weights.index].copy()
    else:
        diff = weights.diff()

    total_turn = diff.abs().sum().sum()
    return total_turn / (len(weights) / periods)

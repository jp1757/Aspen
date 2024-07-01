"""
Calculates statistics on scalar inputs
"""

import numpy as np


def deannualize(rate: float, *, periods: int) -> float:
    """
    De-annualize input rate to new frequency. i.e. convert annual risk-free-rate
    to monthly

    :param rate: (float) input interest rate
    :param periods: (int) frequency to convert to. Set to 12 for monthly,
        252 for daily, 52 weekly etc.
    :return: (float) de-annualized rate
    """
    return np.power(rate + 1, 1 / periods) - 1

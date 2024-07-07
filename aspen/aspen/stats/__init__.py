"""Package for calculating performance statistics, charting etc"""

from aspen.stats.library import portfolio

# Extend pandas objects, tying stats time series functions so that they
# are accessible as easily as pandas.Series.cagr(periods=12) etc
from pandas.core.base import PandasObject
from inspect import getmembers, isfunction

# Loop through all functions in timeseries module
for name, func in getmembers(portfolio, isfunction):
    if not name.startswith("__"):  # Function is not name mangled (aka private)
        setattr(PandasObject, name, func)

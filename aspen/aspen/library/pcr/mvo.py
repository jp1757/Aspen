"""
Mean-variance portfolio optimization implementation
"""

from typing import List

import cvxpy
import numpy as np
import pandas as pd

from aspen.pcr.core import IPortConstruct
from aspen.tform.core import ITForm
import aspen.library.tform.align


def vcv(
    *, dates: pd.DatetimeIndex, asset_tr: pd.DataFrame, freq: str, window: ITForm
) -> pd.DataFrame:
    """
    Build a time series of variance co-variance matrices using a ITForm object to
    define the window moving through time ('rolling', 'ewm', 'expanding' etc...).

    It calculates a VCV foreach date from dates.min to dates.max based on the input
    frequency.  Then re-indexes back to the input dates and fills forward to ensure that
    there is a VCV on every date.

    :param dates: (pd.DatetimeIndex) dates to build VCVs over ensuring that there is a
        VCV on every date
    :param asset_tr: (pd.DataFrame) asset total return index data (i.e. prices) used to
        build VCVs
    :param freq: (str) pandas frequency string used to calculate VCVs
    :param window: (ITForm) transformation to specify rolling window
        i.e. aspen.tform.generic.TForm('rolling', window=20) would calculate VCVs over
        every 20 period rolling window

    :return: (pd.DataFrame) multi-index dataframe containing a VCV on every date.
        Index: date, asset ids
        Columns: asset ids
    """

    # Re-index asset total return index data to frequency for VCV matrix
    tr = (
        aspen.library.tform.align.Reindex(
            pd.date_range(dates.min(), dates.max(), freq=freq), ffill_trailing=True
        )
        .apply(asset_tr)
        .ffill()
    )
    # Build a VCV matrix based on input ITForm window ('rollin', 'ewm' etc)
    returns = tr.pct_change().dropna()
    cov = window.apply(returns).cov()

    # Re-index back to input dates index to ensure there is a VCV on every date.
    # Dates could be daily & VCV frequency could be weekly, then we would just
    # fill forward the weekly VCV for every day in dates
    cov = (
        aspen.library.tform.align.Reindex(dates, ffill_trailing=True)
        .apply(cov.unstack())
        .ffill()
        .stack(dropna=False)
    )

    return cov


def zero_correlations(cov: pd.DataFrame) -> pd.DataFrame:
    _vcv = cov.copy()
    v = np.zeros(_vcv.shape)
    np.fill_diagonal(v, np.diag(_vcv))

    return pd.DataFrame(v, index=_vcv.index, columns=_vcv.columns)


def optimize(
    *,
    cov: pd.DataFrame,
    risk_aversion: float,
    alpha: pd.Series,
    constraints: List = None,
    solver=cvxpy.ECOS
):

    # Define weights variable
    num_assets = len(alpha)
    weights = cvxpy.Variable(num_assets)

    # Target Vol
    target_vol = cvxpy.quad_form(weights, cov.values * (risk_aversion / 2))

    # Target alpha
    target_alpha = alpha.values @ weights

    # Define the objective (maximize something akin to Sharpe)
    objective = cvxpy.Maximize(target_alpha - target_vol)

    # Check Constraints
    constraints = [] if constraints is None else constraints

    # Define the problem
    problem = cvxpy.Problem(objective, constraints)

    # Solve the problem
    problem.solve(solver=solver)

    return weights


class FixedRisk(IPortConstruct):

    def __init__(
        self,
        *,
        cov: pd.DataFrame,
        risk_aversion: float,
        ic: float = 0.1,
        constraints: List = None,
        solver=cvxpy.ECOS
    ) -> None:

        self.cov = cov
        self.risk_aversion = risk_aversion
        self.ic = ic
        self.constraints = constraints
        self.solver = solver
        self._sdate = cov.index.get_level_values(0).min()

    def weights(
        self, *, date: pd.Timestamp, signals: pd.DataFrame, asset: pd.DataFrame
    ) -> pd.Series:

        # Target alpha
        signal = signals.iloc[-1]

        # Check current date vs VCV start date
        if date < self._sdate:
            return pd.Series([np.NaN] * len(signal), index=signal.index, name=date)

        # Get VCV for date & align with alpha assets
        _vcv = self.cov.loc[date].loc[signal.index]

        # Zero out the noisy correlations
        _vcv = zero_correlations(_vcv)

        # Convert signal to alpha
        vol = pd.Series(np.sqrt(np.diag(_vcv)), index=_vcv.columns)
        alpha = self.ic * signal * vol

        # Run optimization
        opt_weights = optimize(
            cov=_vcv,
            risk_aversion=self.risk_aversion,
            alpha=alpha,
            constraints=self.constraints,
            solver=self.solver,
        )

        # Return weights
        return pd.Series(opt_weights.value, index=alpha.index, name=date)

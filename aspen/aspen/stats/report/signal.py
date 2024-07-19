"""
Utility functions for plotting charts & snapshot performance table
"""

from typing import List

import matplotlib.pyplot as plt
import pandas as pd

import aspen.pcr.library.quintile
import aspen.signals.library.normalise
import aspen.stats.library.signal
import aspen.stats.report.portfolio
from aspen.backtest.generic import BTest
from aspen.backtest.portfolio import Portfolio
from aspen.signals.core import ISignal
from aspen.signals.generic import Signals

ID = aspen.stats.report.portfolio.ID


def qport(
        isignal: ISignal, *, asset_tr: pd.DataFrame, bins: int, pct_rank: bool = False
) -> pd.DataFrame:
    # Wrap signal in ISignals object
    _signals = Signals(isignal)
    # Init quantile portfolio construction
    pcr = aspen.pcr.library.quintile.QuantileEW(long_bin=1, short_bin=bins)
    # Normalize signal data cross-sectionally into bins
    normalise = aspen.signals.library.normalise.Quantile(
        rank=aspen.tform.library.rank.RankXSect(pct=pct_rank), bins=bins
    )
    # Init backtest object
    btest = BTest(
        name=isignal.name,
        dates=asset_tr.index,
        tr=asset_tr,
        signals=_signals,
        pcr=pcr,
        normalise=normalise,
        signal=isignal.name
    )
    # Portfolio object for returns calculation
    port = Portfolio(isignal.name, asset_tr=asset_tr, weights=btest.run())

    return port.tr


def ics(signal: ISignal, asset_tr: pd.DataFrame, lags: List[int]):
    # Calculate cross-sectional ICs
    scores = {
        lag: pd.Series(
            aspen.stats.library.signal.ic_xsect(
                tr=asset_tr, signal=signal.calculate(), lag=lag, rank="basic"
            ),
            name=signal.name
        )
        for lag in lags
    }
    # Calculate t-stat scores for each IC
    tstats = {
        lag: aspen.stats.library.signal.tstat(scr) for lag, scr in scores.items()
    }

    return scores, tstats


def series(scores, tstats, name: str) -> pd.DataFrame:
    # Combine stats
    return pd.concat(
        [
            pd.Series(
                {f"IC[{x}]": v.mean() for x, v in scores.items()}, name=name
            ),
            pd.Series(
                {f"TStat[{x}]": v for x, v in tstats.items()}, name=name
            )
        ]
    ).round(2)


def table(*signal: ISignal, asset_tr: pd.DataFrame, lags: List[int]):
    _series = [
        series(
            *ics(signal=s, asset_tr=asset_tr, lags=lags), name=s.name
        )
        for s in signal
    ]
    df = pd.concat(_series, axis=1).T
    df.index.name = ID
    return df.reset_index()


def snapshot(
        *signal: ISignal,
        asset_tr: pd.DataFrame,
        lags: List[int],
        bins: int,
        periods: int,
        rolling: int,
        ic: int = 1,
        pct_rank: bool = False,
        bps: bool = False,
        dateformat: str = "%Y-%m-%d",
) -> None:
    """
    Plot a snapshot of portfolio statistics including a summary table and 4 plots:
    total return, rolling sharpe, drawdown, rolling volatility

    :param tr: (pd.Series) portfolio total return price series i.e [1.0, 1.01, 0.98...]
    :param periods: (int) periods per year i.e. 12 for monthly, 252 for daily etc.
    :param rolling: (int) rolling period for sharpe and volatility plots
    :param bps: (bool, optional) whether to represent values in basis points (True)
        or percentage points (False)
    :param dateformat: (str, optional) string format to represent dates
    :return: None
    """

    # Get signal summary tables
    sig_df = table(*signal, asset_tr=asset_tr, lags=lags)

    # Build backtests for each signal & calculate portfolio returns
    ports = [
        qport(s, asset_tr=asset_tr, bins=bins, pct_rank=pct_rank)
        for s in signal
    ]
    port_df = aspen.stats.report.portfolio.table(
        *ports, periods=periods, bps=bps, dateformat=dateformat
    )

    # Combine summary dataframes
    df = pd.merge(port_df, sig_df, on=ID)

    # Build plot
    fig = plt.figure(figsize=(14, 12))

    gs = fig.add_gridspec(5, 2)

    ax1 = fig.add_subplot(gs[0, :])

    ax6 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[3, 1])
    ax9 = fig.add_subplot(gs[4, 1])

    __xticklabels(ax6, ax4, ax2, ax5, ax3, ax7)

    tbl = ax1.table(cellText=df.values, colLabels=df.columns, loc='center')
    ax1.axis('off')
    # Set the font size
    tbl.auto_set_font_size(False)  # Disable automatic font size adjustment
    tbl.set_fontsize(10)  # Set the desired font size
    tbl.auto_set_column_width(col=list(range(len(df.columns))))  # Adjust all columns

    # Plot total return
    ax2.plot(aspen.stats.report.portfolio.merge(*ports, metric="tr"))
    ax2.set_ylabel("Total Return (%)")
    ax2.grid()

    # Plot Drawdown
    ax3.plot(aspen.stats.report.portfolio.merge(*ports, metric="drawdown"))
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid()

    # Plot Sharpe Ratio
    ax4.plot(
        aspen.stats.report.portfolio.merge(
            *ports, metric="sharpe", rolling=rolling, pct=False, periods=periods
        )
    )
    ax4.set_ylabel("Sharpe")
    ax4.grid()

    # Plot annualised volatility
    vol = aspen.stats.report.portfolio.merge(
        *ports, metric="vol", rolling=rolling, periods=periods
    )
    ax5.plot(vol)
    ax5.set_ylabel("Volatility (%)")
    ax5.grid()

    # Get rolling ICs
    _ics = {
        s.name:
        aspen.stats.report.signal.ics(
            signal=s, asset_tr=asset_tr, lags=list(range(1, 13))
        )[0]
        for s in signal
    }
    rolling_ic = pd.concat([x[ic] for k, x in _ics.items()], axis=1).rolling(rolling).mean()
    ax6.plot(rolling_ic)
    ax6.set_ylabel(f"Rolling[{rolling}] IC")
    ax6.grid()

    # Get IC decay
    _ic_decay = [
        pd.Series(pd.DataFrame(v).mean(), name=x) for x, v in _ics.items()
    ]
    pd.concat(_ic_decay, axis=1).plot(ax=ax7, kind="bar")
    ax7.set_ylabel(f"IC Decay")
    ax7.get_legend().remove()
    ax7.grid()

    # Calculate IC decay success rate
    _srs = {
        k:
            pd.Series(
                {
                    k2: aspen.stats.library.signal.success_rate(v2)
                    for k2, v2 in v.items()
                },
                name=k
            )
        for k, v in _ics.items()
    }
    success_rate = pd.DataFrame(_srs)
    ax9.set_ylabel(f"len(IC > 0) / len(IC)")
    ax9.plot(success_rate)
    ax9.grid()

    # Legend
    fig.legend(vol.columns, loc="lower right")

    fig.tight_layout()

    plt.show()


def __xticklabels(*axis, rotation: int = 30, alignment: str = "right"):
    for ax in axis:
        for label in ax.get_xticklabels():
            label.set_ha(alignment)
            label.set_rotation(rotation)


def __bar(ax, df: pd.DataFrame, width: float = 0.25):
    multiplier = 0

    for attribute, measurement in df.items():
        offset = width * multiplier
        rects = ax.bar(df.index + offset, measurement, label=attribute, width=width)
        multiplier += 1

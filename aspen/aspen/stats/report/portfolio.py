"""
Utility functions for plotting charts & snapshot performance table
"""

import matplotlib.pyplot as plt
import pandas as pd

from aspen.library.tform.align import Align
import aspen.stats.library.signal


ID = "id"


def merge(
        *tr: pd.Series, metric: str, align: bool = True, pct: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Calculate a specific statistic, align and combine into a single dataframe

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :param metric: (str) function to call on input series object.  NB: all statistic
        functions in aspen.stats.library.ts are available along with pandas built in
        functions
    :param align: (bool, optional) whether to align resulting series
    :param pct: (bool, optional) convert values into pct_rank by multiplying by 100
    :param kwargs: (optional) other parameters to pass to target metric function
    :return: pd.Dataframe of combined values for each input series for each metric
    """
    if align:
        tf = Align(*[t.index for t in tr])
        tr = [tf.apply(t) for t in tr]

    tr = [getattr(t, metric)(**kwargs) for t in tr]
    tr = pd.concat(tr, axis=1).ffill()

    if pct:
        return tr * 100
    else:
        return tr


def series(
        tr: pd.Series, *, periods: int, bps: bool = False, dateformat: str = "%Y-%m-%d"
) -> pd.Series:
    """
    Create a series of statistics for an input set of portfolio returns; cagr, volatility,
    sharpe, max drawdown, max drawdown date, kurtosis, skew, 1Y return, 3Y return, 5Y return,
    start date, end date, number of periods,

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :param periods: (int) periods per year i.e. 12 for monthly, 252 for daily etc.
    :param bps: (bool, optional) whether to represent values in basis points (True)
        or percentage points (False)
    :param dateformat: (str, optional) string format to represent dates
    :return: pd.Series of different portfolio statistics
    """
    # Convert to % or bps
    mult = 10000 if bps else 100
    lbl = "bps" if bps else "%"

    # Calculate returns
    returns = tr.pct_change()

    # Drawdown
    dd = tr.drawdown().sort_values()
    dd_min = dd.min()
    assert dd_min == dd.iloc[0]
    dd_date = dd.index[0]

    snap = pd.Series(
        [
            round(tr.cagr(periods=periods) * mult, 1),
            round(tr.vol(periods=periods) * mult, 1),
            round(tr.sharpe(periods=periods), 2),
            round(aspen.stats.library.signal.tstat(returns), 2),
            round(dd_min * mult, 1),
            dd_date.strftime(dateformat),
            round(returns.kurt(), 2),
            round(returns.skew(), 2),
            round(tr.year_return(years=1, periods=periods) * mult, 1),
            round(tr.year_return(years=3, periods=periods) * mult, 1),
            round(tr.year_return(years=5, periods=periods) * mult, 1),
            tr.index.min().strftime(dateformat),
            tr.index.max().strftime(dateformat),
            len(tr),
        ],
        index=[
            f"cagr[{lbl}]", f"vol[{lbl}]", "sharpe", "tstat", f"drawdown[{lbl}]", "drawdown dt",
            "kurtosis", "skew", f"1Y[{lbl}]", f"3Y[{lbl}]", f"5Y[{lbl}]", "sdate",
            "edate", "# periods"
        ],
        name=tr.name
    )

    return snap


def table(
        *tr: pd.Series, periods: int, bps: bool = False, dateformat: str = "%Y-%m-%d"
) -> pd.DataFrame:
    """
    Calculate a combined table of statistics for various portfolios

    :param tr: (pd.Series) total return price series i.e [1.0, 1.01, 0.98...]
    :param periods: (int) periods per year i.e. 12 for monthly, 252 for daily etc.
    :param bps: (bool, optional) whether to represent values in basis points (True)
        or percentage points (False)
    :param dateformat: (str, optional) string format to represent dates
    :return: pd.Dataframe of combined portfolio statistics
    """
    df = pd.concat(
        [series(tr=x, periods=periods, bps=bps, dateformat=dateformat) for x in tr],
        axis=1
    ).T
    df.index.name = ID
    return df.reset_index()


def snapshot(
        *tr: pd.Series,
        periods: int,
        rolling: int,
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
    df = table(*tr, periods=periods, bps=bps, dateformat=dateformat)

    fig = plt.figure(figsize=(14, 8))

    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax4 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax4)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax5 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax5)
    plt.setp(ax3.get_xticklabels(), visible=False)

    tbl = ax1.table(cellText=df.values, colLabels=df.columns, loc='center')
    ax1.axis('off')
    # Set the font size
    tbl.auto_set_font_size(False)  # Disable automatic font size adjustment
    tbl.set_fontsize(10)  # Set the desired font size
    tbl.auto_set_column_width(col=list(range(len(df.columns))))  # Adjust all columns

    ax2.plot(merge(*tr, metric="tr"))
    ax2.set_ylabel("Total Return (%)")
    ax2.grid()

    ax3.plot(merge(*tr, metric="drawdown"))
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid()

    ax4.plot(merge(*tr, metric="sharpe", rolling=rolling, pct=False, periods=periods))
    ax4.set_ylabel("Sharpe")
    ax4.grid()

    vol = merge(*tr, metric="vol", rolling=rolling, periods=periods)
    ax5.plot(vol)
    ax5.set_ylabel("Volatility (%)")
    ax5.grid()

    fig.legend(vol.columns, loc='upper right')

    fig.autofmt_xdate()
    fig.tight_layout()

    plt.show()

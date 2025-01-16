# ASPEN
### Quant Research & Backtesting Framework
Building strategies from the ground up using this simple building-block approach
<br>
<br>
-------------------------------------------
Tests found in `aspen/tests` <br>
Please refer to tests and `aspen.library` for examples on how to use and implement the framework
<br>
-------------------------------------------
To install package, navigate to the `aspen/aspen` directory and run the following command in your python environment:
> pip install --editable . <br>
(make sure to include the trailing '.')
-------------------------------------------

### Flow
- Define transformations by implementing `aspen.tform.core.ITForm`
	- Please note for simple 'out-the-box' transformations (eg. `pandas.mean()`) instead of creating a new object implementing `aspen.tform.core.ITForm` just create an instance of `aspen.tform.generic.TForm` and pass the function name and any additional required parameters via `kwargs`
	- If you need to use multiple functions but it still doesn't warrant its own class (eg. `pandas.rolling(10).mean`) use a transformation pipeline `aspen.tform.pipeline.Pipeline` which accepts multiple `ITForm` objects and is also one itself
- Combine data transformations in a signal object `aspen.signals.core.ISignal` and pass it data (asset prices, fundamental, macro etc)
- Combine multiple signals together - `aspen.signals.core.ISignals` (eg. cross-sectional mean `aspen.library.signals.combine.XSMean`)
- Create a backtest object - interface: `aspen.backtest.core.IBTest`, generic implementation: `aspen.backtest.generic.BTest`
	- Pass asset total return price data
	- Pass `ISignals` object
	- Define and pass a portfolio construction object `aspen.pcr.core.IPortConstruct` (for examples see `aspen.library.pcr`)
	- Option to define and pass a normalisation object which is applied to the signals data before running the backtest eg. z-score - `aspen.signals.core.INormalise`
	- Specify a rebalance object `aspen.rebalance.core.IRebal`, default to rebalance on all dates
	- Run the backtest to generate a `pd.DataFrame` of weights
- Create an instance of a portfolio object - `aspen.backtest.portfolio.Portfolio`
	- Pass weights
	- Option to pass FX information if assets do not share the same base currency
	- Extra: the `aspen.backtest.portfolio.drift` function will apply drift to a set of input weights, useful for more accurate turnover calculations ie. calculate the intra-month daily drift from monthly weights
	- Retrieve portfolio returns, total return prices, and asset returns
- Use `aspen.stats` package to calculate performance statistics at both the portfolio and signal level
	- note: once you import `aspen.stats`, all functions in `aspen.stats.portolio` are exposed to `pandas.Series` objects eg. `pd.Series().cagr(periods=12)` or `pd.Series().vol(periods=252)` etc

<br>
-------------------------------------------

### To-dos
- Create an additional `IBTest` implementation that does not iterate dates
	> For simpler non-optimization portfolio construction methodologies it will be significantly faster to not iterate through dates eg. equal-weight, market cap etc
import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Working with Time Series
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Pandas was originally developed in the context of financial modeling, so as you might expect, it contains an extensive set of tools for working with dates, times, and time-indexed data.
        Date and time data comes in a few flavors, which we will discuss here:

        - *Timestamps* reference particular moments in time (e.g., July 4th, 2021 at 7:00 a.m.).
        - *Time intervals* and *periods* reference a length of time between a particular beginning and end point; for example, the month of June 2021. Periods usually reference a special case of time intervals in which each interval is of uniform length and does not overlap (e.g., 24-hour-long periods comprising days).
        - *Time deltas* or *durations* reference an exact length of time (e.g., a duration of 22.56 seconds).

        This chapter will introduce how to work with each of these types of date/time data in Pandas.
        This is by no means a complete guide to the time series tools available in Python or Pandas, but instead is intended as a broad overview of how you as a user should approach working with time series.
        We will start with a brief discussion of tools for dealing with dates and times in Python, before moving more specifically to a discussion of the tools provided by Pandas.
        After listing some resources that go into more depth, we will review some short examples of working with time series data in Pandas.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Dates and Times in Python

        The Python world has a number of available representations of dates, times, deltas, and time spans.
        While the time series tools provided by Pandas tend to be the most useful for data science applications, it is helpful to see their relationship to other tools used in Python.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Native Python Dates and Times: datetime and dateutil

        Python's basic objects for working with dates and times reside in the built-in `datetime` module.
        Along with the third-party `dateutil` module, you can use this to quickly perform a host of useful functionalities on dates and times.
        For example, you can manually build a date using the `datetime` type:
        """
    )
    return


@app.cell
def _():
    from datetime import datetime
    datetime(year=2021, month=7, day=4)
    return (datetime,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or, using the `dateutil` module, you can parse dates from a variety of string formats:
        """
    )
    return


@app.cell
def _():
    from dateutil import parser
    date = parser.parse("4th of July, 2021")
    date
    return date, parser


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Once you have a `datetime` object, you can do things like printing the day of the week:
        """
    )
    return


@app.cell
def _(date):
    date.strftime('%A')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here we've used one of the standard string format codes for printing dates (`'%A'`), which you can read about in the [`strftime` section](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) of Python's [`datetime` documentation](https://docs.python.org/3/library/datetime.html).
        Documentation of other useful date utilities can be found in [``dateutil``'s online documentation](http://labix.org/python-dateutil).
        A related package to be aware of is [`pytz`](http://pytz.sourceforge.net/), which contains tools for working with the most migraine-inducing element of time series data: time zones.

        The power of `datetime` and `dateutil` lies in their flexibility and easy syntax: you can use these objects and their built-in methods to easily perform nearly any operation you might be interested in.
        Where they break down is when you wish to work with large arrays of dates and times:
        just as lists of Python numerical variables are suboptimal compared to NumPy-style typed numerical arrays, lists of Python `datetime` objects are suboptimal compared to typed arrays of encoded dates.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Typed Arrays of Times: NumPy's datetime64

        NumPy's `datetime64` dtype encodes dates as 64-bit integers, and thus allows arrays of dates to be represented compactly and operated on in an efficient manner.
        The `datetime64` requires a specific input format:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    date_1 = np.array('2021-07-04', dtype=np.datetime64)
    date_1
    return date_1, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Once we have dates in this form, we can quickly do vectorized operations on it:
        """
    )
    return


@app.cell
def _(date_1, np):
    date_1 + np.arange(12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because of the uniform type in NumPy `datetime64` arrays, this kind of operation can be accomplished much more quickly than if we were working directly with Python's `datetime` objects, especially as arrays get large
        (we introduced this type of vectorization in [Computation on NumPy Arrays: Universal Functions](02.03-Computation-on-arrays-ufuncs.ipynb)).

        One detail of the `datetime64` and related `timedelta64` objects is that they are built on a *fundamental time unit*.
        Because the `datetime64` object is limited to 64-bit precision, the range of encodable times is $2^{64}$ times this fundamental unit.
        In other words, `datetime64` imposes a trade-off between *time resolution* and *maximum time span*.

        For example, if you want a time resolution of 1 nanosecond, you only have enough information to encode a range of $2^{64}$ nanoseconds, or just under 600 years.
        NumPy will infer the desired unit from the input; for example, here is a day-based `datetime`:
        """
    )
    return


@app.cell
def _(np):
    np.datetime64('2021-07-04')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here is a minute-based datetime:
        """
    )
    return


@app.cell
def _(np):
    np.datetime64('2021-07-04 12:00')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can force any desired fundamental unit using one of many format codes; for example, here we'll force a nanosecond-based time:
        """
    )
    return


@app.cell
def _(np):
    np.datetime64('2021-07-04 12:59:59.50', 'ns')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The following table, drawn from the NumPy `datetime64` documentation, lists the available format codes along with the relative and absolute time spans that they can encode:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        |Code  | Meaning     | Time span (relative) | Time span (absolute)   |
        |------|-------------|----------------------|------------------------|
        | `Y`  | Year        | ± 9.2e18 years       | [9.2e18 BC, 9.2e18 AD] |
        | `M`  | Month       | ± 7.6e17 years       | [7.6e17 BC, 7.6e17 AD] |
        | `W`  | Week        | ± 1.7e17 years       | [1.7e17 BC, 1.7e17 AD] |
        | `D`  | Day         | ± 2.5e16 years       | [2.5e16 BC, 2.5e16 AD] |
        | `h`  | Hour        | ± 1.0e15 years       | [1.0e15 BC, 1.0e15 AD] |
        | `m`  | Minute      | ± 1.7e13 years       | [1.7e13 BC, 1.7e13 AD] |
        | `s`  | Second      | ± 2.9e12 years       | [ 2.9e9 BC, 2.9e9 AD]  |
        | `ms` | Millisecond | ± 2.9e9 years        | [ 2.9e6 BC, 2.9e6 AD]  |
        | `us` | Microsecond | ± 2.9e6 years        | [290301 BC, 294241 AD] |
        | `ns` | Nanosecond  | ± 292 years          | [ 1678 AD, 2262 AD]    |
        | `ps` | Picosecond  | ± 106 days           | [ 1969 AD, 1970 AD]    |
        | `fs` | Femtosecond | ± 2.6 hours          | [ 1969 AD, 1970 AD]    |
        | `as` | Attosecond  | ± 9.2 seconds        | [ 1969 AD, 1970 AD]    |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For the types of data we see in the real world, a useful default is `datetime64[ns]`, as it can encode a useful range of modern dates with a suitably fine precision.

        Finally, note that while the `datetime64` data type addresses some of the deficiencies of the built-in Python `datetime` type, it lacks many of the convenient methods and functions provided by `datetime` and especially `dateutil`.
        More information can be found in [NumPy's `datetime64` documentation](http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Dates and Times in Pandas: The Best of Both Worlds

        Pandas builds upon all the tools just discussed to provide a `Timestamp` object, which combines the ease of use of `datetime` and `dateutil` with the efficient storage and vectorized interface of `numpy.datetime64`.
        From a group of these `Timestamp` objects, Pandas can construct a `DatetimeIndex` that can be used to index data in a `Series` or `DataFrame`.

        For example, we can use Pandas tools to repeat the demonstration from earlier.
        We can parse a flexibly formatted string date and use format codes to output the day of the week, as follows:
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    date_2 = pd.to_datetime('4th of July, 2021')
    date_2
    return date_2, pd


@app.cell
def _(date_2):
    date_2.strftime('%A')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Additionally, we can do NumPy-style vectorized operations directly on this same object:
        """
    )
    return


@app.cell
def _(date_2, np, pd):
    date_2 + pd.to_timedelta(np.arange(12), 'D')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the next section, we will take a closer look at manipulating time series data with the tools provided by Pandas.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Pandas Time Series: Indexing by Time

        The Pandas time series tools really become useful when you begin to index data by timestamps.
        For example, we can construct a `Series` object that has time-indexed data:
        """
    )
    return


@app.cell
def _(pd):
    index = pd.DatetimeIndex(['2020-07-04', '2020-08-04',
                              '2021-07-04', '2021-08-04'])
    data = pd.Series([0, 1, 2, 3], index=index)
    data
    return data, index


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And now that we have this data in a `Series`, we can make use of any of the `Series` indexing patterns we discussed in previous chapters, passing values that can be coerced into dates:
        """
    )
    return


@app.cell
def _(data):
    data['2020-07-04':'2021-07-04']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are additional special date-only indexing operations, such as passing a year to obtain a slice of all data from that year:
        """
    )
    return


@app.cell
def _(data):
    data['2021']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Later, we will see additional examples of the convenience of dates-as-indices.
        But first, let's take a closer look at the available time series data structures.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Pandas Time Series Data Structures

        This section will introduce the fundamental Pandas data structures for working with time series data:

        - For *timestamps*, Pandas provides the `Timestamp` type. As mentioned before, this is essentially a replacement for Python's native `datetime`, but it's based on the more efficient `numpy.datetime64` data type. The associated `Index` structure is `DatetimeIndex`.
        - For *time periods*, Pandas provides the `Period` type. This encodes a fixed-frequency interval based on `numpy.datetime64`. The associated index structure is `PeriodIndex`.
        - For *time deltas* or *durations*, Pandas provides the `Timedelta` type. `Timedelta` is a more efficient replacement for Python's native `datetime.timedelta` type, and is based on `numpy.timedelta64`. The associated index structure is `TimedeltaIndex`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The most fundamental of these date/time objects are the `Timestamp` and `DatetimeIndex` objects.
        While these class objects can be invoked directly, it is more common to use the `pd.to_datetime` function, which can parse a wide variety of formats.
        Passing a single date to `pd.to_datetime` yields a `Timestamp`; passing a series of dates by default yields a `DatetimeIndex`, as you can see here:
        """
    )
    return


@app.cell
def _(datetime, pd):
    dates = pd.to_datetime([datetime(2021, 7, 3), '4th of July, 2021',
                           '2021-Jul-6', '07-07-2021', '20210708'])
    dates
    return (dates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Any `DatetimeIndex` can be converted to a `PeriodIndex` with the `to_period` function, with the addition of a frequency code; here we'll use `'D'` to indicate daily frequency:
        """
    )
    return


@app.cell
def _(dates):
    dates.to_period('D')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A `TimedeltaIndex` is created, for example, when a date is subtracted from another:
        """
    )
    return


@app.cell
def _(dates):
    dates - dates[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Regular Sequences: pd.date_range

        To make creation of regular date sequences more convenient, Pandas offers a few functions for this purpose: `pd.date_range` for timestamps, `pd.period_range` for periods, and `pd.timedelta_range` for time deltas.
        We've seen that Python's `range` and NumPy's `np.arange` take a start point, end point, and optional step size and return a sequence.
        Similarly, `pd.date_range` accepts a start date, an end date, and an optional frequency code to create a regular sequence of dates:
        """
    )
    return


@app.cell
def _(pd):
    pd.date_range('2015-07-03', '2015-07-10')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Alternatively, the date range can be specified not with a start and end point, but with a start point and a number of periods:
        """
    )
    return


@app.cell
def _(pd):
    pd.date_range('2015-07-03', periods=8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The spacing can be modified by altering the `freq` argument, which defaults to `D`.
        For example, here we construct a range of hourly timestamps:
        """
    )
    return


@app.cell
def _(pd):
    pd.date_range('2015-07-03', periods=8, freq='H')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To create regular sequences of `Period` or `Timedelta` values, the similar `pd.period_range` and `pd.timedelta_range` functions are useful.
        Here are some monthly periods:
        """
    )
    return


@app.cell
def _(pd):
    pd.period_range('2015-07', periods=8, freq='M')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And a sequence of durations increasing by an hour:
        """
    )
    return


@app.cell
def _(pd):
    pd.timedelta_range(0, periods=6, freq='H')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        All of these require an understanding of Pandas frequency codes, which are summarized in the next section.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Frequencies and Offsets

        Fundamental to these Pandas time series tools is the concept of a *frequency* or *date offset*. The following table summarizes the main codes available; as with the `D` (day) and `H` (hour) codes demonstrated in the previous sections, we can use these to specify any desired frequency spacing:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        | Code | Description       | Code | Description          |
        |------|-------------------|------|----------------------|
        | `D`  | Calendar day      | `B`  | Business day         |
        | `W`  | Weekly            |      |                      |
        | `M`  | Month end         | `BM` | Business month end   |
        | `Q`  | Quarter end       | `BQ` | Business quarter end |
        | `A`  | Year end          | `BA` | Business year end    |
        | `H`  | Hours             | `BH` | Business hours       |
        | `T`  | Minutes           |      |                      |
        | `S`  | Seconds           |      |                      |
        | `L`  | Milliseconds       |      |                      |
        | `U`  | Microseconds      |      |                      |
        | `N`  | Nanoseconds       |      |                      |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The monthly, quarterly, and annual frequencies are all marked at the end of the specified period.
        Adding an `S` suffix to any of these causes them to instead be marked at the beginning:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        | Code  | Description       | Code  | Description            |
        |-------|-------------------|-------|------------------------|
        | `MS`  | Month start       |`BMS`  | Business month start   |
        | `QS`  | Quarter start     |`BQS`  | Business quarter start |
        | `AS`  | Year start        |`BAS`  | Business year start    |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Additionally, you can change the month used to mark any quarterly or annual code by adding a three-letter month code as a suffix:

        - `Q-JAN`, `BQ-FEB`, `QS-MAR`, `BQS-APR`, etc.
        - `A-JAN`, `BA-FEB`, `AS-MAR`, `BAS-APR`, etc.

        In the same way, the split point of the weekly frequency can be modified by adding a three-letter weekday code:

        - `W-SUN`, `W-MON`, `W-TUE`, `W-WED`, etc.

        On top of this, codes can be combined with numbers to specify other frequencies.
        For example, for a frequency of 2 hours and 30 minutes, we can combine the hour (`H`) and minute (`T`) codes as follows:
        """
    )
    return


@app.cell
def _(pd):
    pd.timedelta_range(0, periods=6, freq="2H30T")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        All of these short codes refer to specific instances of Pandas time series offsets, which can be found in the `pd.tseries.offsets` module.
        For example, we can create a business day offset directly as follows:
        """
    )
    return


@app.cell
def _(pd):
    from pandas.tseries.offsets import BDay
    pd.date_range('2015-07-01', periods=6, freq=BDay())
    return (BDay,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more discussion of the use of frequencies and offsets, see the [`DateOffset` section](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects) of the Pandas documentation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Resampling, Shifting, and Windowing

        The ability to use dates and times as indices to intuitively organize and access data is an important aspect of the Pandas time series tools.
        The benefits of indexed data in general (automatic alignment during operations, intuitive data slicing and access, etc.) still apply, and Pandas provides several additional time series–specific operations.

        We will take a look at a few of those here, using some stock price data as an example.
        Because Pandas was developed largely in a finance context, it includes some very specific tools for financial data.
        For example, the accompanying `pandas-datareader` package (installable via `pip install pandas-datareader`) knows how to import data from various online sources.
        Here we will load part of the S&P 500 price history:
        """
    )
    return


@app.cell
def _(data_1):
    from pandas_datareader import data
    sp500 = data_1.DataReader('^GSPC', start='2018', end='2022', data_source='yahoo')
    sp500.head()
    return data, sp500


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For simplicity, we'll use just the closing price:
        """
    )
    return


@app.cell
def _(sp500):
    sp500_1 = sp500['Close']
    return (sp500_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can visualize this using the ``plot`` method, after the normal Matplotlib setup boilerplate (see [Part 4](04.00-Introduction-To-Matplotlib.ipynb)); the result is shown in the following figure:
        """
    )
    return


@app.cell
def _():
    # "%matplotlib inline\nimport matplotlib.pyplot as plt\nplt.style.use('seaborn-whitegrid')\nsp500.plot();" command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Resampling and Converting Frequencies

        One common need when dealing with time series data is resampling at a higher or lower frequency.
        This can be done using the `resample` method, or the much simpler `asfreq` method.
        The primary difference between the two is that `resample` is fundamentally a *data aggregation*, while `asfreq` is fundamentally a *data selection*.

        Let's compare what the two return when we downsample the S&P 500 closing price data.
        Here we will resample the data at the end of business year; the following figure shows the result:
        """
    )
    return


@app.cell
def _(plt, sp500_1):
    sp500_1.plot(alpha=0.5, style='-')
    sp500_1.resample('BA').mean().plot(style=':')
    sp500_1.asfreq('BA').plot(style='--')
    plt.legend(['input', 'resample', 'asfreq'], loc='upper left')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice the difference: at each point, `resample` reports the *average of the previous year*, while `asfreq` reports the *value at the end of the year*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For upsampling, `resample` and `asfreq` are largely equivalent, though `resample` has many more options available.
        In this case, the default for both methods is to leave the upsampled points empty; that is, filled with NA values.
        Like the `pd.fillna` function discussed in [Handling Missing Data](03.04-Missing-Values.ipynb), `asfreq` accepts a `method` argument to specify how values are imputed.
        Here, we will resample the business day data at a daily frequency (i.e., including weekends); the following figure shows the result:
        """
    )
    return


@app.cell
def _(plt, sp500_1):
    _fig, _ax = plt.subplots(2, sharex=True)
    data_2 = sp500_1.iloc[:20]
    data_2.asfreq('D').plot(ax=_ax[0], marker='o')
    data_2.asfreq('D', method='bfill').plot(ax=_ax[1], style='-o')
    data_2.asfreq('D', method='ffill').plot(ax=_ax[1], style='--o')
    _ax[1].legend(['back-fill', 'forward-fill'])
    return (data_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because the S&P 500 data only exists for business days, the top panel has gaps representing NA values.
        The bottom panel shows the differences between two strategies for filling the gaps: forward filling and backward filling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Time Shifts

        Another common time series–specific operation is shifting of data in time.
        For this, Pandas provides the `shift` method, which can be used to shift data by a given number of entries.
        With time series data sampled at a regular frequency, this can give us a way to explore trends over time.

        For example, here we resample the data to daily values, and shift by 364 to compute the 1-year return on investment for the S&P 500 over time (see the following figure):
        """
    )
    return


@app.cell
def _(plt, sp500_1):
    sp500_2 = sp500_1.asfreq('D', method='pad')
    ROI = 100 * (sp500_2.shift(-365) - sp500_2) / sp500_2
    ROI.plot()
    plt.ylabel('% Return on Investment after 1 year')
    return ROI, sp500_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The worst one-year return was around March 2019, with the coronavirus-related market crash exactly a year later. As you might expect, the best one-year return was to be found in March 2020, for those with enough foresight or luck to buy low.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Rolling Windows

        Calculating rolling statistics is a third type of time series–specific operation implemented by Pandas.
        This can be accomplished via the `rolling` attribute of `Series` and `DataFrame` objects, which returns a view similar to what we saw with the `groupby` operation (see [Aggregation and Grouping](03.08-Aggregation-and-Grouping.ipynb)).
        This rolling view makes available a number of aggregation operations by default.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        For example, we can look at the one-year centered rolling mean and standard deviation of the  stock prices (see the following figure):
        """
    )
    return


@app.cell
def _(pd, sp500_2):
    rolling = sp500_2.rolling(365, center=True)
    data_3 = pd.DataFrame({'input': sp500_2, 'one-year rolling_mean': rolling.mean(), 'one-year rolling_median': rolling.median()})
    _ax = data_3.plot(style=['-', '--', ':'])
    _ax.lines[0].set_alpha(0.3)
    return data_3, rolling


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As with `groupby` operations, the `aggregate` and `apply` methods can be used for custom rolling computations.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Where to Learn More

        This chapter has provided only a brief summary of some of the most essential features of time series tools provided by Pandas; for a more complete discussion, you can refer to the ["Time Series/Date Functionality" section](http://pandas.pydata.org/pandas-docs/stable/timeseries.html) of the Pandas online documentation.

        Another excellent resource is the book [*Python for Data Analysis*](https://learning.oreilly.com/library/view/python-for-data/9781098104023/) by Wes McKinney (O'Reilly).
        It is an invaluable resource on the use of Pandas.
        In particular, this book emphasizes time series tools in the context of business and finance, and focuses much more on particular details of business calendars, time zones, and related topics.

        As always, you can also use the IPython help functionality to explore and try out further options available to the functions and methods discussed here. I find this often is the best way to learn a new Python tool.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: Visualizing Seattle Bicycle Counts

        As a more involved example of working with time series data, let's take a look at bicycle counts on Seattle's [Fremont Bridge](http://www.openstreetmap.org/#map=17/47.64813/-122.34965).
        This data comes from an automated bicycle counter installed in late 2012, which has inductive sensors on the east and west sidewalks of the bridge.
        The hourly bicycle counts can be downloaded from [http://data.seattle.gov](http://data.seattle.gov); the Fremont Bridge Bicycle Counter dataset is available under the Transportation category.

        The CSV used for this book can be downloaded as follows:
        """
    )
    return


@app.cell
def _():
    # url = ('https://raw.githubusercontent.com/jakevdp/'
    #        'bicycle-data/main/FremontBridge.csv')
    # !curl -O {url}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Once this dataset is downloaded, we can use Pandas to read the CSV output into a `DataFrame`.
        We will specify that we want the `Date` column as an index, and we want these dates to be automatically parsed:
        """
    )
    return


@app.cell
def _(pd):
    data_4 = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
    data_4.head()
    return (data_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For convenience, we'll shorten the column names:
        """
    )
    return


@app.cell
def _(data_4):
    data_4.columns = ['Total', 'East', 'West']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's take a look at the summary statistics for this data:
        """
    )
    return


@app.cell
def _(data_4):
    data_4.dropna().describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Visualizing the Data

        We can gain some insight into the dataset by visualizing it.
        Let's start by plotting the raw data (see the following figure):
        """
    )
    return


@app.cell
def _(data_4, plt):
    data_4.plot()
    plt.ylabel('Hourly Bicycle Count')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The ~150,000 hourly samples are far too dense for us to make much sense of.
        We can gain more insight by resampling the data to a coarser grid.
        Let's resample by week (see the following figure):
        """
    )
    return


@app.cell
def _(data_4, plt):
    weekly = data_4.resample('W').sum()
    weekly.plot(style=['-', ':', '--'])
    plt.ylabel('Weekly bicycle count')
    return (weekly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This reveals some trends: as you might expect, people bicycle more in the summer than in the winter, and even within a particular season the bicycle use varies from week to week (likely dependent on weather; see [In Depth: Linear Regression](05.06-Linear-Regression.ipynb), where we explore this further). Further, the effect of the COVID-19 pandemic on commuting patterns is quite clear, starting in early 2020.

        Another option that comes in handy for aggregating the data is to use a rolling mean, utilizing the `pd.rolling_mean` function.
        Here we'll examine the 30-day rolling mean of our data, making sure to center the window (see the following figure):
        """
    )
    return


@app.cell
def _(data_4, plt):
    daily = data_4.resample('D').sum()
    daily.rolling(30, center=True).sum().plot(style=['-', ':', '--'])
    plt.ylabel('mean hourly count')
    return (daily,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The jaggedness of the result is due to the hard cutoff of the window.
        We can get a smoother version of a rolling mean using a window function—for example, a Gaussian window, as shown in the following figure.
        The following code specifies both the width of the window (here, 50 days) and the width of the Gaussian window (here, 10 days):
        """
    )
    return


@app.cell
def _(daily):
    daily.rolling(50, center=True,
                  win_type='gaussian').sum(std=10).plot(style=['-', ':', '--']);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Digging into the Data

        While these smoothed data views are useful to get an idea of the general trend in the data, they hide much of the structure.
        For example, we might want to look at the average traffic as a function of the time of day.
        We can do this using the `groupby` functionality discussed in [Aggregation and Grouping](03.08-Aggregation-and-Grouping.ipynb) (see the following figure):
        """
    )
    return


@app.cell
def _(data_4, np):
    by_time = data_4.groupby(data_4.index.time).mean()
    hourly_ticks = 4 * 60 * 60 * np.arange(6)
    by_time.plot(xticks=hourly_ticks, style=['-', ':', '--'])
    return by_time, hourly_ticks


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The hourly traffic is a strongly bimodal sequence, with peaks around 8:00 a.m. and 5:00 p.m.
        This is likely evidence of a strong component of commuter traffic crossing the bridge.
        There is a directional component as well: according to the data, the east sidewalk is used more during the a.m. commute, and the west sidewalk is used more during the p.m. commute.

        We also might be curious about how things change based on the day of the week. Again, we can do this with a simple `groupby` (see the following figure):
        """
    )
    return


@app.cell
def _(data_4):
    by_weekday = data_4.groupby(data_4.index.dayofweek).mean()
    by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
    by_weekday.plot(style=['-', ':', '--'])
    return (by_weekday,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This shows a strong distinction between weekday and weekend totals, with around twice as many average riders crossing the bridge on Monday through Friday than on Saturday and Sunday.

        With this in mind, let's do a compound `groupby` and look at the hourly trends on weekdays versus weekends.
        We'll start by grouping by flags marking the weekend and the time of day:
        """
    )
    return


@app.cell
def _(data_4, np):
    weekend = np.where(data_4.index.weekday < 5, 'Weekday', 'Weekend')
    by_time_1 = data_4.groupby([weekend, data_4.index.time]).mean()
    return by_time_1, weekend


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we'll use some of the Matplotlib tools that will be described in [Multiple Subplots](04.08-Multiple-Subplots.ipynb) to plot two panels side by side, as shown in the following figure:
        """
    )
    return


@app.cell
def _(by_time_1, hourly_ticks):
    import matplotlib.pyplot as plt
    _fig, _ax = plt.subplots(1, 2, figsize=(14, 5))
    by_time_1.loc['Weekday'].plot(ax=_ax[0], title='Weekdays', xticks=hourly_ticks, style=['-', ':', '--'])
    by_time_1.loc['Weekend'].plot(ax=_ax[1], title='Weekends', xticks=hourly_ticks, style=['-', ':', '--'])
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The result shows a bimodal commuting pattern during the work week, and a unimodal recreational pattern during the weekends.
        It might be interesting to dig through this data in more detail and examine the effects of weather, temperature, time of year, and other factors on people's commuting patterns; for further discussion, see my blog post ["Is Seattle Really Seeing an Uptick in Cycling?"](https://jakevdp.github.io/blog/2014/06/10/is-seattle-really-seeing-an-uptick-in-cycling/), which uses a subset of this data.
        We will also revisit this dataset in the context of modeling in [In Depth: Linear Regression](05.06-Linear-Regression.ipynb).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

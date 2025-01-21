import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Aggregations: min, max, and Everything in Between""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A first step in exploring any dataset is often to compute various summary statistics.
        Perhaps the most common summary statistics are the mean and standard deviation, which allow you to summarize the "typical" values in a dataset, but other aggregations are useful as well (the sum, product, median, minimum and maximum, quantiles, etc.).

        NumPy has fast built-in aggregation functions for working on arrays; we'll discuss and try out some of them here.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summing the Values in an Array

        As a quick example, consider computing the sum of all values in an array.
        Python itself can do this using the built-in `sum` function:
        """
    )
    return


@app.cell
def _():
    import numpy as np

    rng = np.random.default_rng()
    return np, rng


@app.cell
def _(rng):
    L = rng.random(100)
    sum(L)
    return (L,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The syntax is quite similar to that of NumPy's `sum` function, and the result is the same in the simplest case:""")
    return


@app.cell
def _(L, np):
    np.sum(L)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""However, because it executes the operation in compiled code, NumPy's version of the operation is computed much more quickly:""")
    return


@app.cell
def _(np, rng):
    big_array = rng.random(1000000)
    sum(big_array)
    np.sum(big_array)
    return (big_array,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Be careful, though: the `sum` function and the `np.sum` function are not identical, which can sometimes lead to confusion!
        In particular, their optional arguments have different meanings (`sum(x, 1)` initializes the sum at `1`, while `np.sum(x, 1)` sums along axis `1`), and `np.sum` is aware of multiple array dimensions, as we will see in the following section.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Minimum and Maximum

        Similarly, Python has built-in `min` and `max` functions, used to find the minimum value and maximum value of any given array:
        """
    )
    return


@app.cell
def _(big_array):
    min(big_array), max(big_array)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""NumPy's corresponding functions have similar syntax, and again operate much more quickly:""")
    return


@app.cell
def _(big_array, np):
    np.min(big_array), np.max(big_array)
    return


@app.cell
def _(big_array, np):
    # magic command not supported in marimo; please file an issue to add support
    min(big_array)
    np.min(big_array)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For `min`, `max`, `sum`, and several other NumPy aggregates, a shorter syntax is to use methods of the array object itself:""")
    return


@app.cell
def _(big_array):
    print(big_array.min(), big_array.max(), big_array.sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Whenever possible, make sure that you are using the NumPy version of these aggregates when operating on NumPy arrays!""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Multidimensional Aggregates

        One common type of aggregation operation is an aggregate along a row or column.
        Say you have some data stored in a two-dimensional array:
        """
    )
    return


@app.cell
def _(rng):
    M = rng.integers(0, 10, (3, 4))
    print(M)
    return (M,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""NumPy aggregations will apply across all elements of a multidimensional array:""")
    return


@app.cell
def _(M):
    M.sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Aggregation functions take an additional argument specifying the *axis* along which the aggregate is computed. For example, we can find the minimum value within each column by specifying `axis=0`:""")
    return


@app.cell
def _(M):
    M.min(axis=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The function returns four values, corresponding to the four columns of numbers.

        Similarly, we can find the maximum value within each row:
        """
    )
    return


@app.cell
def _(M):
    M.max(axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The way the axis is specified here can be confusing to users coming from other languages.
        The `axis` keyword specifies the dimension of the array that will be *collapsed*, rather than the dimension that will be returned.
        So, specifying `axis=0` means that axis 0 will be collapsed: for two-dimensional arrays, values within each column will be aggregated.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Other Aggregation Functions

        NumPy provides several other aggregation functions with a similar API, and additionally most have a `NaN`-safe counterpart that computes the result while ignoring missing values, which are marked by the special IEEE floating-point `NaN` value (see [Handling Missing Data](03.04-Missing-Values.ipynb)).

        The following table provides a list of useful aggregation functions available in NumPy:

        |Function name    |   NaN-safe version| Description                                   |
        |-----------------|-------------------|-----------------------------------------------|
        | `np.sum`        | `np.nansum`       | Compute sum of elements                       |
        | `np.prod`       | `np.nanprod`      | Compute product of elements                   |
        | `np.mean`       | `np.nanmean`      | Compute mean of elements                      |
        | `np.std`        | `np.nanstd`       | Compute standard deviation                    |
        | `np.var`        | `np.nanvar`       | Compute variance                              |
        | `np.min`        | `np.nanmin`       | Find minimum value                            |
        | `np.max`        | `np.nanmax`       | Find maximum value                            |
        | `np.argmin`     | `np.nanargmin`    | Find index of minimum value                   |
        | `np.argmax`     | `np.nanargmax`    | Find index of maximum value                   |
        | `np.median`     | `np.nanmedian`    | Compute median of elements                    |
        | `np.percentile` | `np.nanpercentile`| Compute rank-based statistics of elements     |
        | `np.any`        | N/A               | Evaluate whether any elements are true        |
        | `np.all`        | N/A               | Evaluate whether all elements are true        |

        You will see these aggregates often throughout the rest of the book.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Example: What Is the Average Height of US Presidents?""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Aggregates available in NumPy can act as summary statistics for a set of values.
        As a small example, let's consider the heights of all US presidents.
        This data is available in the file *president_heights.csv*, which is a comma-separated list of labels and values:
        """
    )
    return


@app.cell
def _():
    # !head -4 data/president_heights.csv
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We'll use the Pandas package, which we'll explore more fully in [Part 3](03.00-Introduction-to-Pandas.ipynb), to read the file and extract this information (note that the heights are measured in centimeters):""")
    return


@app.cell
def _(np):
    import pandas as pd

    data = pd.read_csv("data/president_heights.csv")
    heights = np.array(data["height(cm)"])
    print(heights)
    return data, heights, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now that we have this data array, we can compute a variety of summary statistics:""")
    return


@app.cell
def _(heights):
    print("Mean height:       ", heights.mean())
    print("Standard deviation:", heights.std())
    print("Minimum height:    ", heights.min())
    print("Maximum height:    ", heights.max())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that in each case, the aggregation operation reduced the entire array to a single summarizing value, which gives us information about the distribution of values.
        We may also wish to compute quantiles:
        """
    )
    return


@app.cell
def _(heights, np):
    print("25th percentile:   ", np.percentile(heights, 25))
    print("Median:            ", np.median(heights))
    print("75th percentile:   ", np.percentile(heights, 75))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that the median height of US presidents is 182 cm, or just shy of six feet.

        Of course, sometimes it's more useful to see a visual representation of this data, which we can accomplish using tools in Matplotlib (we'll discuss Matplotlib more fully in [Part 4](04.00-Introduction-To-Matplotlib.ipynb)). For example, this code generates the following chart:
        """
    )
    return


@app.cell
def _():
    # "%matplotlib inline\nimport matplotlib.pyplot as plt\nplt.style.use('seaborn-whitegrid')" command supported automatically in marimo
    return


@app.cell
def _(heights, plt):
    plt.hist(heights)
    plt.title("Height Distribution of US Presidents")
    plt.xlabel("height (cm)")
    plt.ylabel("number")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

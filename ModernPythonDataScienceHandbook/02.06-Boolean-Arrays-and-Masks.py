import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Comparisons, Masks, and Boolean Logic""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This chapter covers the use of Boolean masks to examine and manipulate values within NumPy arrays.
        Masking comes up when you want to extract, modify, count, or otherwise manipulate values in an array based on some criterion: for example, you might wish to count all values greater than a certain value, or remove all outliers that are above some threshold.
        In NumPy, Boolean masking is often the most efficient way to accomplish these types of tasks.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: Counting Rainy Days

        Imagine you have a series of data that represents the amount of precipitation each day for a year in a given city.
        For example, here we'll load the daily rainfall statistics for the city of Seattle in 2015, using Pandas (see [Part 3](03.00-Introduction-to-Pandas.ipynb)):
        """
    )
    return


@app.cell
def _():
    import numpy as np
    from vega_datasets import data

    # Use DataFrame operations to extract rainfall as a NumPy array
    rainfall_mm = np.array(
        data.seattle_weather().set_index("date")["precipitation"]["2015"]
    )
    len(rainfall_mm)
    return data, np, rainfall_mm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The array contains 365 values, giving daily rainfall in millimeters from January 1 to December 31, 2015.

        As a first quick visualization, let's look at the histogram of rainy days in the following figure, which was generated using Matplotlib (we will explore this tool more fully in [Part 4](04.00-Introduction-To-Matplotlib.ipynb)):
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    # plt.style.use('seaborn-whitegrid')
    return (plt,)


@app.cell
def _(plt, rainfall_mm):
    plt.hist(rainfall_mm, 40)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This histogram gives us a general idea of what the data looks like: despite the city's rainy reputation, the vast majority of days in Seattle saw near zero measured rainfall in 2015.
        But this doesn't do a good job of conveying some information we'd like to see: for example, how many rainy days were there in the year? What was the average precipitation on those rainy days? How many days were there with more than 10 mm of rainfall?

        One approach to this would be to answer these questions by hand: we could loop through the data, incrementing a counter each time we see values in some desired range.
        But for reasons discussed throughout this chapter, such an approach is very inefficient from the standpoint of both time writing code and time computing the result.
        We saw in [Computation on NumPy Arrays: Universal Functions](02.03-Computation-on-arrays-ufuncs.ipynb) that NumPy's ufuncs can be used in place of loops to do fast element-wise arithmetic operations on arrays; in the same way, we can use other ufuncs to do element-wise *comparisons* over arrays, and we can then manipulate the results to answer the questions we have.
        We'll leave the data aside for now, and discuss some general tools in NumPy to use *masking* to quickly answer these types of questions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Comparison Operators as Ufuncs

        [Computation on NumPy Arrays: Universal Functions](02.03-Computation-on-arrays-ufuncs.ipynb) introduced ufuncs, and focused in particular on arithmetic operators. We saw that using `+`, `-`, `*`, `/`, and other operators on arrays leads to element-wise operations.
        NumPy also implements comparison operators such as `<` (less than) and `>` (greater than) as element-wise ufuncs.
        The result of these comparison operators is always an array with a Boolean data type.
        All six of the standard comparison operations are available:
        """
    )
    return


@app.cell
def _(np):
    x = np.array([1, 2, 3, 4, 5])
    return (x,)


@app.cell
def _(x):
    x < 3  # less than
    return


@app.cell
def _(x):
    x > 3  # greater than
    return


@app.cell
def _(x):
    x <= 3  # less than or equal
    return


@app.cell
def _(x):
    x >= 3  # greater than or equal
    return


@app.cell
def _(x):
    x != 3  # not equal
    return


@app.cell
def _(x):
    x == 3  # equal
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""It is also possible to do an element-wise comparison of two arrays, and to include compound expressions:""")
    return


@app.cell
def _(x):
    (2 * x) == (x**2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As in the case of arithmetic operators, the comparison operators are implemented as ufuncs in NumPy; for example, when you write `x < 3`, internally NumPy uses `np.less(x, 3)`.
            A summary of the comparison operators and their equivalent ufuncs is shown here:

        | Operator    | Equivalent ufunc  | Operator   | Equivalent ufunc |
        |-------------|-------------------|------------|------------------|
        |`==`         |`np.equal`         |`!=`        |`np.not_equal`    |
        |`<`          |`np.less`          |`<=`        |`np.less_equal`   |
        |`>`          |`np.greater`       |`>=`        |`np.greater_equal`|
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Just as in the case of arithmetic ufuncs, these will work on arrays of any size and shape.
        Here is a two-dimensional example:
        """
    )
    return


@app.cell
def _(np):
    rng = np.random.default_rng(seed=1701)
    x_1 = rng.integers(10, size=(3, 4))
    x_1
    return rng, x_1


@app.cell
def _(x_1):
    x_1 < 6
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In each case, the result is a Boolean array, and NumPy provides a number of straightforward patterns for working with these Boolean results.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Working with Boolean Arrays

        Given a Boolean array, there are a host of useful operations you can do.
        We'll work with `x`, the two-dimensional array we created earlier:
        """
    )
    return


@app.cell
def _(x_1):
    print(x_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Counting Entries

        To count the number of `True` entries in a Boolean array, `np.count_nonzero` is useful:
        """
    )
    return


@app.cell
def _(np, x_1):
    np.count_nonzero(x_1 < 6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that there are eight array entries that are less than 6.
        Another way to get at this information is to use `np.sum`; in this case, `False` is interpreted as `0`, and `True` is interpreted as `1`:
        """
    )
    return


@app.cell
def _(np, x_1):
    np.sum(x_1 < 6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The benefit of `np.sum` is that, like with other NumPy aggregation functions, this summation can be done along rows or columns as well:""")
    return


@app.cell
def _(np, x_1):
    np.sum(x_1 < 6, axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This counts the number of values less than 6 in each row of the matrix.

        If we're interested in quickly checking whether any or all the values are `True`, we can use (you guessed it) `np.any` or `np.all`:
        """
    )
    return


@app.cell
def _(np, x_1):
    np.any(x_1 > 8)
    return


@app.cell
def _(np, x_1):
    np.any(x_1 < 0)
    return


@app.cell
def _(np, x_1):
    np.all(x_1 < 10)
    return


@app.cell
def _(np, x_1):
    np.all(x_1 == 6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`np.all` and `np.any` can be used along particular axes as well. For example:""")
    return


@app.cell
def _(np, x_1):
    np.all(x_1 < 8, axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here all the elements in the third row are less than 8, while this is not the case for others.

        Finally, a quick warning: as mentioned in [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb), Python has built-in `sum`, `any`, and `all` functions. These have a different syntax than the NumPy versions, and in particular will fail or produce unintended results when used on multidimensional arrays. Be sure that you are using `np.sum`, `np.any`, and `np.all` for these examples!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Boolean Operators

        We've already seen how we might count, say, all days with less than 20 mm of rain, or all days with more than 10 mm of rain.
        But what if we want to know how many days there were with more than 10 mm and less than 20 mm of rain? We can accomplish this with Python's *bitwise logic operators*, `&`, `|`, `^`, and `~`.
        Like with the standard arithmetic operators, NumPy overloads these as ufuncs that work element-wise on (usually Boolean) arrays.

        For example, we can address this sort of compound question as follows:
        """
    )
    return


@app.cell
def _(np, rainfall_mm):
    np.sum((rainfall_mm > 10) & (rainfall_mm < 20))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This tells us that there were 16 days with rainfall of between 10 and 20 millimeters.

        The parentheses here are important. Because of operator precedence rules, with the parentheses removed this expression would be evaluated as follows, which results in an error:

        ``` python
        rainfall_mm > (10 & rainfall_mm) < 20
        ```

        Let's demonstrate a more complicated expression. Using De Morgan's laws, we can compute the same result in a different manner:
        """
    )
    return


@app.cell
def _(np, rainfall_mm):
    np.sum(~((rainfall_mm <= 10) | (rainfall_mm >= 20)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Combining comparison operators and Boolean operators on arrays can lead to a wide range of efficient logical operations.

        The following table summarizes the bitwise Boolean operators and their equivalent ufuncs:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        | Operator    | Equivalent ufunc  | Operator    | Equivalent ufunc  |
        |-------------|-------------------|-------------|-------------------|
        |`&`          |`np.bitwise_and`   |&#124;       |`np.bitwise_or`    |
        |`^`          |`np.bitwise_xor`   |`~`          |`np.bitwise_not`   |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using these tools, we can start to answer many of the questions we might have about our weather data.
        Here are some examples of results we can compute when combining masking with aggregations:
        """
    )
    return


@app.cell
def _(np, rainfall_mm):
    print("Number days without rain:  ", np.sum(rainfall_mm == 0))
    print("Number days with rain:     ", np.sum(rainfall_mm != 0))
    print("Days with more than 10 mm: ", np.sum(rainfall_mm > 10))
    print(
        "Rainy days with < 5 mm:    ", np.sum((rainfall_mm > 0) & (rainfall_mm < 5))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Boolean Arrays as Masks

        In the preceding section we looked at aggregates computed directly on Boolean arrays.
        A more powerful pattern is to use Boolean arrays as masks, to select particular subsets of the data themselves. Let's return to our `x` array from before:
        """
    )
    return


@app.cell
def _(x_1):
    x_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Suppose we want an array of all values in the array that are less than, say, 5. We can obtain a Boolean array for this condition easily, as we've already seen:""")
    return


@app.cell
def _(x_1):
    x_1 < 5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, to *select* these values from the array, we can simply index on this Boolean array; this is known as a *masking* operation:""")
    return


@app.cell
def _(x_1):
    x_1[x_1 < 5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        What is returned is a one-dimensional array filled with all the values that meet this condition; in other words, all the values in positions at which the mask array is `True`.

        We are then free to operate on these values as we wish.
        For example, we can compute some relevant statistics on our Seattle rain data:
        """
    )
    return


@app.cell
def _(np, rainfall_mm):
    # construct a mask of all rainy days
    rainy = rainfall_mm > 0

    # construct a mask of all summer days (June 21st is the 172nd day)
    days = np.arange(365)
    summer = (days > 172) & (days < 262)

    print(
        "Median precip on rainy days in 2015 (mm):   ",
        np.median(rainfall_mm[rainy]),
    )
    print(
        "Median precip on summer days in 2015 (mm):  ",
        np.median(rainfall_mm[summer]),
    )
    print(
        "Maximum precip on summer days in 2015 (mm): ", np.max(rainfall_mm[summer])
    )
    print(
        "Median precip on non-summer rainy days (mm):",
        np.median(rainfall_mm[rainy & ~summer]),
    )
    return days, rainy, summer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""By combining Boolean operations, masking operations, and aggregates, we can very quickly answer these sorts of questions about our dataset.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using the Keywords and/or Versus the Operators &/|

        One common point of confusion is the difference between the keywords `and` and `or` on the one hand, and the operators `&` and `|` on the other.
        When would you use one versus the other?

        The difference is this: `and` and `or` operate on the object as a whole, while `&` and `|` operate on the elements within the object.

        When you use `and` or `or`, it is equivalent to asking Python to treat the object as a single Boolean entity.
        In Python, all nonzero integers will evaluate as `True`. Thus:
        """
    )
    return


@app.cell
def _():
    bool(42), bool(0)
    return


@app.cell
def _():
    bool(42 and 0)
    return


@app.cell
def _():
    bool(42 or 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""When you use `&` and `|` on integers, the expression operates on the bitwise representation of the element, applying the *and* or the *or* to the individual bits making up the number:""")
    return


@app.cell
def _():
    bin(42)
    return


@app.cell
def _():
    bin(59)
    return


@app.cell
def _():
    bin(42 & 59)
    return


@app.cell
def _():
    bin(42 | 59)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that the corresponding bits of the binary representation are compared in order to yield the result.

        When you have an array of Boolean values in NumPy, this can be thought of as a string of bits where `1 = True` and `0 = False`, and `&` and `|` will operate similarly to in the preceding examples:
        """
    )
    return


@app.cell
def _(np):
    A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
    A | B
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""But if you use `or` on these arrays it will try to evaluate the truth or falsehood of the entire array object, which is not a well-defined value:""")
    return


@app.cell
def _(A, B):
    A or B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Similarly, when evaluating a Boolean expression on a given array, you should use `|` or `&` rather than `or` or `and`:""")
    return


@app.cell
def _(np):
    x_2 = np.arange(10)
    (x_2 > 4) & (x_2 < 8)
    return (x_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Trying to evaluate the truth or falsehood of the entire array will give the same `ValueError` we saw previously:""")
    return


@app.cell
def _(x_2):
    x_2 > 4 and x_2 < 8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So, remember this: `and` and `or` perform a single Boolean evaluation on an entire object, while `&` and `|` perform multiple Boolean evaluations on the content (the individual bits or bytes) of an object.
        For Boolean NumPy arrays, the latter is nearly always the desired operation.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

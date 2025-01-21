import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # High-Performance Pandas: eval and query
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As we've already seen in previous chapters, the power of the PyData stack is built upon the ability of NumPy and Pandas to push basic operations into lower-level compiled code via an intuitive higher-level syntax: examples are vectorized/broadcasted operations in NumPy, and grouping-type operations in Pandas.
        While these abstractions are efficient and effective for many common use cases, they often rely on the creation of temporary intermediate objects, which can cause undue overhead in computational time and memory use.

        To address this, Pandas includes some methods that allow you to directly access C-speed operations without costly allocation of intermediate arrays: `eval` and `query`, which rely on the [NumExpr package](https://github.com/pydata/numexpr).
        In this chapter I will walk you through their use and give some rules of thumb about when you might think about using them.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Motivating query and eval: Compound Expressions

        We've seen previously that NumPy and Pandas support fast vectorized operations; for example, when adding the elements of two arrays:
        """
    )
    return


app._unparsable_cell(
    r"""
    import numpy as np
    rng = np.random.default_rng(42)
    x = rng.random(1000000)
    y = rng.random(1000000)
    %timeit x + y
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As discussed in [Computation on NumPy Arrays: Universal Functions](02.03-Computation-on-arrays-ufuncs.ipynb), this is much faster than doing the addition via a Python loop or comprehension:
        """
    )
    return


app._unparsable_cell(
    r"""
    # magic command not supported in marimo; please file an issue to add support
    # %timeit np.fromiter((xi + yi for xi, yi in zip(x, y)),
                        dtype=x.dtype, count=len(x))
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But this abstraction can become less efficient when computing compound expressions.
        For example, consider the following expression:
        """
    )
    return


@app.cell
def _(x, y):
    mask = (x > 0.5) & (y < 0.5)
    return (mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because NumPy evaluates each subexpression, this is roughly equivalent to the following:
        """
    )
    return


@app.cell
def _(x, y):
    tmp1 = x > 0.5
    tmp2 = y < 0.5
    mask_1 = tmp1 & tmp2
    return mask_1, tmp1, tmp2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In other words, *every intermediate step is explicitly allocated in memory*. If the `x` and `y` arrays are very large, this can lead to significant memory and computational overhead.
        The NumExpr library gives you the ability to compute this type of compound expression element by element, without the need to allocate full intermediate arrays.
        The [NumExpr documentation](https://github.com/pydata/numexpr) has more details, but for the time being it is sufficient to say that the library accepts a *string* giving the NumPy-style expression you'd like to compute:
        """
    )
    return


@app.cell
def _(mask_1, np):
    import numexpr
    mask_numexpr = numexpr.evaluate('(x > 0.5) & (y < 0.5)')
    np.all(mask_1 == mask_numexpr)
    return mask_numexpr, numexpr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The benefit here is that NumExpr evaluates the expression in a way that avoids temporary arrays where possible, and thus can be much more efficient than NumPy, especially for long sequences of computations on large arrays.
        The Pandas `eval` and `query` tools that we will discuss here are conceptually similar, and are essentially Pandas-specific wrappers of NumExpr functionality.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## pandas.eval for Efficient Operations

        The `eval` function in Pandas uses string expressions to efficiently compute operations on `DataFrame` objects.
        For example, consider the following data:
        """
    )
    return


@app.cell
def _(rng):
    import pandas as pd
    nrows, ncols = 100000, 100
    df1, df2, df3, df4 = (pd.DataFrame(rng.random((nrows, ncols)))
                          for i in range(4))
    return df1, df2, df3, df4, ncols, nrows, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To compute the sum of all four ``DataFrame``s using the typical Pandas approach, we can just write the sum:
        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %timeit df1 + df2 + df3 + df4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The same result can be computed via ``pd.eval`` by constructing the expression as a string:
        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %timeit pd.eval('df1 + df2 + df3 + df4')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `eval` version of this expression is about 50% faster (and uses much less memory), while giving the same result:
        """
    )
    return


@app.cell
def _(df1, df2, df3, df4, np, pd):
    np.allclose(df1 + df2 + df3 + df4,
                pd.eval('df1 + df2 + df3 + df4'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        `pd.eval` supports a wide range of operations.
        To demonstrate these, we'll use the following integer data:
        """
    )
    return


@app.cell
def _(pd, rng):
    df1_1, df2_1, df3_1, df4_1, df5 = (pd.DataFrame(rng.integers(0, 1000, (100, 3))) for i in range(5))
    return df1_1, df2_1, df3_1, df4_1, df5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Arithmetic operators
        `pd.eval` supports all arithmetic operators. For example:
        """
    )
    return


@app.cell
def _(df1_1, df2_1, df3_1, df4_1, df5, np, pd):
    result1 = -df1_1 * df2_1 / (df3_1 + df4_1) - df5
    result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
    np.allclose(result1, result2)
    return result1, result2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Comparison operators
        `pd.eval` supports all comparison operators, including chained expressions:
        """
    )
    return


@app.cell
def _(df1_1, df2_1, df3_1, df4_1, np, pd):
    result1_1 = (df1_1 < df2_1) & (df2_1 <= df3_1) & (df3_1 != df4_1)
    result2_1 = pd.eval('df1 < df2 <= df3 != df4')
    np.allclose(result1_1, result2_1)
    return result1_1, result2_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Bitwise operators
        `pd.eval` supports the `&` and `|` bitwise operators:
        """
    )
    return


@app.cell
def _(df1_1, df2_1, df3_1, df4_1, np, pd):
    result1_2 = (df1_1 < 0.5) & (df2_1 < 0.5) | (df3_1 < df4_1)
    result2_2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
    np.allclose(result1_2, result2_2)
    return result1_2, result2_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In addition, it supports the use of the literal `and` and `or` in Boolean expressions:
        """
    )
    return


@app.cell
def _(np, pd, result1_2):
    result3 = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
    np.allclose(result1_2, result3)
    return (result3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Object attributes and indices

        `pd.eval` supports access to object attributes via the `obj.attr` syntax and indexes via the `obj[index]` syntax:
        """
    )
    return


@app.cell
def _(df2_1, df3_1, np, pd):
    result1_3 = df2_1.T[0] + df3_1.iloc[1]
    result2_3 = pd.eval('df2.T[0] + df3.iloc[1]')
    np.allclose(result1_3, result2_3)
    return result1_3, result2_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Other operations

        Other operations, such as function calls, conditional statements, loops, and other more involved constructs are currently *not* implemented in `pd.eval`.
        If you'd like to execute these more complicated types of expressions, you can use the NumExpr library itself.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## DataFrame.eval for Column-Wise Operations

        Just as Pandas has a top-level `pd.eval` function, `DataFrame` objects have an `eval` method that works in similar ways.
        The benefit of the `eval` method is that columns can be referred to by name.
        We'll use this labeled array as an example:
        """
    )
    return


@app.cell
def _(pd, rng):
    df = pd.DataFrame(rng.random((1000, 3)), columns=['A', 'B', 'C'])
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using `pd.eval` as in the previous section, we can compute expressions with the three columns like this:
        """
    )
    return


@app.cell
def _(df, np, pd):
    result1_4 = (df['A'] + df['B']) / (df['C'] - 1)
    result2_4 = pd.eval('(df.A + df.B) / (df.C - 1)')
    np.allclose(result1_4, result2_4)
    return result1_4, result2_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `DataFrame.eval` method allows much more succinct evaluation of expressions with the columns:
        """
    )
    return


@app.cell
def _(df, np, result1_4):
    result3_1 = df.eval('(A + B) / (C - 1)')
    np.allclose(result1_4, result3_1)
    return (result3_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice here that we treat *column names as variables* within the evaluated expression, and the result is what we would wish.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Assignment in DataFrame.eval

        In addition to the options just discussed, `DataFrame.eval`  also allows assignment to any column.
        Let's use the `DataFrame` from before, which has columns `'A'`, `'B'`, and `'C'`:
        """
    )
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can use `df.eval` to create a new column `'D'` and assign to it a value computed from the other columns:
        """
    )
    return


@app.cell
def _(df):
    df.eval('D = (A + B) / C', inplace=True)
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the same way, any existing column can be modified:
        """
    )
    return


@app.cell
def _(df):
    df.eval('D = (A - B) / C', inplace=True)
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Local Variables in DataFrame.eval

        The `DataFrame.eval` method supports an additional syntax that lets it work with local Python variables.
        Consider the following:
        """
    )
    return


@app.cell
def _(df, np):
    column_mean = df.mean(1)
    result1_5 = df['A'] + column_mean
    result2_5 = df.eval('A + @column_mean')
    np.allclose(result1_5, result2_5)
    return column_mean, result1_5, result2_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `@` character here marks a *variable name* rather than a *column name*, and lets you efficiently evaluate expressions involving the two "namespaces": the namespace of columns, and the namespace of Python objects.
        Notice that this `@` character is only supported by the `DataFrame.eval` *method*, not by the `pandas.eval` *function*, because the `pandas.eval` function only has access to the one (Python) namespace.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The DataFrame.query Method

        The `DataFrame` has another method based on evaluated strings, called `query`.
        Consider the following:
        """
    )
    return


@app.cell
def _(df, np, pd):
    result1_6 = df[(df.A < 0.5) & (df.B < 0.5)]
    result2_6 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
    np.allclose(result1_6, result2_6)
    return result1_6, result2_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As with the example used in our discussion of `DataFrame.eval`, this is an expression involving columns of the `DataFrame`.
        However, it cannot be expressed using the `DataFrame.eval` syntax!
        Instead, for this type of filtering operation, you can use the `query` method:
        """
    )
    return


@app.cell
def _(df, np, result1_6):
    result2_7 = df.query('A < 0.5 and B < 0.5')
    np.allclose(result1_6, result2_7)
    return (result2_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In addition to being a more efficient computation, compared to the masking expression this is much easier to read and understand.
        Note that the `query` method also accepts the `@` flag to mark local variables:
        """
    )
    return


@app.cell
def _(df, np):
    Cmean = df['C'].mean()
    result1_7 = df[(df.A < Cmean) & (df.B < Cmean)]
    result2_8 = df.query('A < @Cmean and B < @Cmean')
    np.allclose(result1_7, result2_8)
    return Cmean, result1_7, result2_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Performance: When to Use These Functions

        When considering whether to use `eval` and `query`, there are two considerations: *computation time* and *memory use*.
        Memory use is the most predictable aspect. As already mentioned, every compound expression involving NumPy arrays or Pandas ``DataFrame``s will result in implicit creation of temporary arrays. For example, this:
        """
    )
    return


@app.cell
def _(df):
    x = df[(df.A < 0.5) & (df.B < 0.5)]
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        is roughly equivalent to this:
        """
    )
    return


@app.cell
def _(df):
    tmp1_1 = df.A < 0.5
    tmp2_1 = df.B < 0.5
    tmp3 = tmp1_1 & tmp2_1
    x_1 = df[tmp3]
    return tmp1_1, tmp2_1, tmp3, x_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If the size of the temporary ``DataFrame``s is significant compared to your available system memory (typically several gigabytes), then it's a good idea to use an `eval` or `query` expression.
        You can check the approximate size of your array in bytes using this:
        """
    )
    return


@app.cell
def _(df):
    df.values.nbytes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        On the performance side, `eval` can be faster even when you are not maxing out your system memory.
        The issue is how your temporary objects compare to the size of the L1 or L2 CPU cache on your system (typically a few megabytes); if they are much bigger, then `eval` can avoid some potentially slow movement of values between the different memory caches.
        In practice, I find that the difference in computation time between the traditional methods and the `eval`/`query` method is usually not significant—if anything, the traditional method is faster for smaller arrays!
        The benefit of `eval`/`query` is mainly in the saved memory, and the sometimes cleaner syntax they offer.

        We've covered most of the details of `eval` and `query` here; for more information on these, you can refer to the Pandas documentation.
        In particular, different parsers and engines can be specified for running these queries; for details on this, see the discussion within the ["Enhancing Performance" section](https://pandas.pydata.org/pandas-docs/dev/user_guide/enhancingperf.html) of the documentation.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

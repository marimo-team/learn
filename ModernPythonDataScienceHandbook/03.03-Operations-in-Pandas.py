import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Operating on Data in Pandas
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One of the strengths of NumPy is that it allows us to perform quick element-wise operations, both with basic arithmetic (addition, subtraction, multiplication, etc.) and with more complicated operations (trigonometric functions, exponential and logarithmic functions, etc.).
        Pandas inherits much of this functionality from NumPy, and the ufuncs introduced in [Computation on NumPy Arrays: Universal Functions](02.03-Computation-on-arrays-ufuncs.ipynb) are key to this.

        Pandas includes a couple of useful twists, however: for unary operations like negation and trigonometric functions, these ufuncs will *preserve index and column labels* in the output, and for binary operations such as addition and multiplication, Pandas will automatically *align indices* when passing the objects to the ufunc.
        This means that keeping the context of data and combining data from different sources—both potentially error-prone tasks with raw NumPy arrays—become essentially foolproof with Pandas.
        We will additionally see that there are well-defined operations between one-dimensional `Series` structures and two-dimensional `DataFrame` structures.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ufuncs: Index Preservation

        Because Pandas is designed to work with NumPy, any NumPy ufunc will work on Pandas `Series` and `DataFrame` objects.
        Let's start by defining a simple `Series` and `DataFrame` on which to demonstrate this:
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    return np, pd


@app.cell
def _(np, pd):
    rng = np.random.default_rng(42)
    ser = pd.Series(rng.integers(0, 10, 4))
    ser
    return rng, ser


@app.cell
def _(pd, rng):
    df = pd.DataFrame(rng.integers(0, 10, (3, 4)),
                      columns=['A', 'B', 'C', 'D'])
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we apply a NumPy ufunc on either of these objects, the result will be another Pandas object *with the indices preserved:*
        """
    )
    return


@app.cell
def _(np, ser):
    np.exp(ser)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is true also for more involved sequences of operations:
        """
    )
    return


@app.cell
def _(df, np):
    np.sin(df * np.pi / 4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Any of the ufuncs discussed in [Computation on NumPy Arrays: Universal Functions](02.03-Computation-on-arrays-ufuncs.ipynb) can be used in a similar manner.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ufuncs: Index Alignment

        For binary operations on two `Series` or `DataFrame` objects, Pandas will align indices in the process of performing the operation.
        This is very convenient when working with incomplete data, as we'll see in some of the examples that follow.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Index Alignment in Series

        As an example, suppose we are combining two different data sources and wish to find only the top three US states by *area* and the top three US states by *population*:
        """
    )
    return


@app.cell
def _(pd):
    area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                      'California': 423967}, name='area')
    population = pd.Series({'California': 39538223, 'Texas': 29145505,
                            'Florida': 21538187}, name='population')
    return area, population


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's see what happens when we divide these to compute the population density:
        """
    )
    return


@app.cell
def _(area, population):
    population / area
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The resulting array contains the *union* of indices of the two input arrays, which could be determined directly from these indices:
        """
    )
    return


@app.cell
def _(area, population):
    area.index.union(population.index)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Any item for which one or the other does not have an entry is marked with `NaN`, or "Not a Number," which is how Pandas marks missing data (see further discussion of missing data in [Handling Missing Data](03.04-Missing-Values.ipynb)).
        This index matching is implemented this way for any of Python's built-in arithmetic expressions; any missing values are marked by `NaN`:
        """
    )
    return


@app.cell
def _(pd):
    A = pd.Series([2, 4, 6], index=[0, 1, 2])
    B = pd.Series([1, 3, 5], index=[1, 2, 3])
    A + B
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If using `NaN` values is not the desired behavior, the fill value can be modified using appropriate object methods in place of the operators.
        For example, calling ``A.add(B)`` is equivalent to calling ``A + B``, but allows optional explicit specification of the fill value for any elements in ``A`` or ``B`` that might be missing:
        """
    )
    return


@app.cell
def _(A, B):
    A.add(B, fill_value=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Index Alignment in DataFrames

        A similar type of alignment takes place for *both* columns and indices when performing operations on `DataFrame` objects:
        """
    )
    return


@app.cell
def _(pd, rng):
    A_1 = pd.DataFrame(rng.integers(0, 20, (2, 2)), columns=['a', 'b'])
    A_1
    return (A_1,)


@app.cell
def _(pd, rng):
    B_1 = pd.DataFrame(rng.integers(0, 10, (3, 3)), columns=['b', 'a', 'c'])
    B_1
    return (B_1,)


@app.cell
def _(A_1, B_1):
    A_1 + B_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that indices are aligned correctly irrespective of their order in the two objects, and indices in the result are sorted.
        As was the case with `Series`, we can use the associated object's arithmetic methods and pass any desired `fill_value` to be used in place of missing entries.
        Here we'll fill with the mean of all values in `A`:
        """
    )
    return


@app.cell
def _(A_1, B_1):
    A_1.add(B_1, fill_value=A_1.values.mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The following table lists Python operators and their equivalent Pandas object methods:

        | Python operator | Pandas method(s)                |
        |-----------------|---------------------------------|
        | `+`             | `add`                           |
        | `-`             | `sub`, `subtract`               |
        | `*`             | `mul`, `multiply`               |
        | `/`             | `truediv`, `div`, `divide`      |
        | `//`            | `floordiv`                      |
        | `%`             | `mod`                           |
        | `**`            | `pow`                           |

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ufuncs: Operations Between DataFrames and Series

        When performing operations between a `DataFrame` and a `Series`, the index and column alignment is similarly maintained, and the result is similar to operations between a two-dimensional and one-dimensional NumPy array.
        Consider one common operation, where we find the difference of a two-dimensional array and one of its rows:
        """
    )
    return


@app.cell
def _(rng):
    A_2 = rng.integers(10, size=(3, 4))
    A_2
    return (A_2,)


@app.cell
def _(A_2):
    A_2 - A_2[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        According to NumPy's broadcasting rules (see [Computation on Arrays: Broadcasting](02.05-Computation-on-arrays-broadcasting.ipynb)), subtraction between a two-dimensional array and one of its rows is applied row-wise.

        In Pandas, the convention similarly operates row-wise by default:
        """
    )
    return


@app.cell
def _(A_2, pd):
    df_1 = pd.DataFrame(A_2, columns=['Q', 'R', 'S', 'T'])
    df_1 - df_1.iloc[0]
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you would instead like to operate column-wise, you can use the object methods mentioned earlier, while specifying the `axis` keyword:
        """
    )
    return


@app.cell
def _(df_1):
    df_1.subtract(df_1['R'], axis=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that these `DataFrame`/`Series` operations, like the operations discussed previously, will automatically align  indices between the two elements:
        """
    )
    return


@app.cell
def _(df_1):
    halfrow = df_1.iloc[0, ::2]
    halfrow
    return (halfrow,)


@app.cell
def _(df_1, halfrow):
    df_1 - halfrow
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This preservation and alignment of indices and columns means that operations on data in Pandas will always maintain the data context, which prevents the common errors that might arise when working with heterogeneous and/or misaligned data in raw NumPy arrays.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Introducing Pandas Objects""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        At a very basic level, Pandas objects can be thought of as enhanced versions of NumPy structured arrays in which the rows and columns are identified with labels rather than simple integer indices.
        As we will see during the course of this chapter, Pandas provides a host of useful tools, methods, and functionality on top of the basic data structures, but nearly everything that follows will require an understanding of what these structures are.
        Thus, before we go any further, let's take a look at these three fundamental Pandas data structures: the `Series`, `DataFrame`, and `Index`.

        We will start our code sessions with the standard NumPy and Pandas imports:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    return np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Pandas Series Object

        A Pandas `Series` is a one-dimensional array of indexed data.
        It can be created from a list or array as follows:
        """
    )
    return


@app.cell
def _(pd):
    data = pd.Series([0.25, 0.5, 0.75, 1.0])
    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `Series` combines a sequence of values with an explicit sequence of indices, which we can access with the `values` and `index` attributes.
        The `values` are simply a familiar NumPy array:
        """
    )
    return


@app.cell
def _(data):
    data.values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `index` is an array-like object of type `pd.Index`, which we'll discuss in more detail momentarily:""")
    return


@app.cell
def _(data):
    data.index
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Like with a NumPy array, data can be accessed by the associated index via the familiar Python square-bracket notation:""")
    return


@app.cell
def _(data):
    data[1]
    return


@app.cell
def _(data):
    data[1:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As we will see, though, the Pandas `Series` is much more general and flexible than the one-dimensional NumPy array that it emulates.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Series as Generalized NumPy Array""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        From what we've seen so far, the `Series` object may appear to be basically interchangeable with a one-dimensional NumPy array.
        The essential difference is that while the NumPy array has an *implicitly defined* integer index used to access the values, the Pandas `Series` has an *explicitly defined* index associated with the values.

        This explicit index definition gives the `Series` object additional capabilities. For example, the index need not be an integer, but can consist of values of any desired type.
        So, if we wish, we can use strings as an index:
        """
    )
    return


@app.cell
def _(pd):
    data_1 = pd.Series([0.25, 0.5, 0.75, 1.0], index=["a", "b", "c", "d"])
    data_1
    return (data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And the item access works as expected:""")
    return


@app.cell
def _(data_1):
    data_1["b"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can even use noncontiguous or nonsequential indices:""")
    return


@app.cell
def _(pd):
    data_2 = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 5, 3, 7])
    data_2
    return (data_2,)


@app.cell
def _(data_2):
    data_2[5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Series as Specialized Dictionary

        In this way, you can think of a Pandas `Series` a bit like a specialization of a Python dictionary.
        A dictionary is a structure that maps arbitrary keys to a set of arbitrary values, and a `Series` is a structure that maps typed keys to a set of typed values.
        This typing is important: just as the type-specific compiled code behind a NumPy array makes it more efficient than a Python list for certain operations, the type information of a Pandas `Series` makes it more efficient than Python dictionaries for certain operations.

        The `Series`-as-dictionary analogy can be made even more clear by constructing a `Series` object directly from a Python dictionary, here the five most populous US states according to the 2020 census:
        """
    )
    return


@app.cell
def _(pd):
    population_dict = {
        "California": 39538223,
        "Texas": 29145505,
        "Florida": 21538187,
        "New York": 20201249,
        "Pennsylvania": 13002700,
    }
    population = pd.Series(population_dict)
    population
    return population, population_dict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""From here, typical dictionary-style item access can be performed:""")
    return


@app.cell
def _(population):
    population["California"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Unlike a dictionary, though, the `Series` also supports array-style operations such as slicing:""")
    return


@app.cell
def _(population):
    population["California":"Florida"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We'll discuss some of the quirks of Pandas indexing and slicing in [Data Indexing and Selection](03.02-Data-Indexing-and-Selection.ipynb).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Constructing Series Objects

        We've already seen a few ways of constructing a Pandas `Series` from scratch. All of them are some version of the following:

        ```python
        pd.Series(data, index=index)
        ```

        where `index` is an optional argument, and `data` can be one of many entities.

        For example, `data` can be a list or NumPy array, in which case `index` defaults to an integer sequence:
        """
    )
    return


@app.cell
def _(pd):
    pd.Series([2, 4, 6])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Or `data` can be a scalar, which is repeated to fill the specified index:""")
    return


@app.cell
def _(pd):
    pd.Series(5, index=[100, 200, 300])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Or it can be a dictionary, in which case `index` defaults to the dictionary keys:""")
    return


@app.cell
def _(pd):
    pd.Series({2: "a", 1: "b", 3: "c"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In each case, the index can be explicitly set to control the order or the subset of keys used:""")
    return


@app.cell
def _(pd):
    pd.Series({2: "a", 1: "b", 3: "c"}, index=[1, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Pandas DataFrame Object

        The next fundamental structure in Pandas is the `DataFrame`.
        Like the `Series` object discussed in the previous section, the `DataFrame` can be thought of either as a generalization of a NumPy array, or as a specialization of a Python dictionary.
        We'll now take a look at each of these perspectives.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### DataFrame as Generalized NumPy Array
        If a `Series` is an analog of a one-dimensional array with explicit indices, a `DataFrame` is an analog of a two-dimensional array with explicit row and column indices.
        Just as you might think of a two-dimensional array as an ordered sequence of aligned one-dimensional columns, you can think of a `DataFrame` as a sequence of aligned `Series` objects.
        Here, by "aligned" we mean that they share the same index.

        To demonstrate this, let's first construct a new `Series` listing the area of each of the five states discussed in the previous section (in square kilometers):
        """
    )
    return


@app.cell
def _(pd):
    area_dict = {
        "California": 423967,
        "Texas": 695662,
        "Florida": 170312,
        "New York": 141297,
        "Pennsylvania": 119280,
    }
    area = pd.Series(area_dict)
    area
    return area, area_dict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now that we have this along with the `population` Series from before, we can use a dictionary to construct a single two-dimensional object containing this information:""")
    return


@app.cell
def _(area, pd, population):
    states = pd.DataFrame({"population": population, "area": area})
    states
    return (states,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Like the `Series` object, the `DataFrame` has an `index` attribute that gives access to the index labels:""")
    return


@app.cell
def _(states):
    states.index
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Additionally, the `DataFrame` has a `columns` attribute, which is an `Index` object holding the column labels:""")
    return


@app.cell
def _(states):
    states.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Thus the `DataFrame` can be thought of as a generalization of a two-dimensional NumPy array, where both the rows and columns have a generalized index for accessing the data.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### DataFrame as Specialized Dictionary

        Similarly, we can also think of a `DataFrame` as a specialization of a dictionary.
        Where a dictionary maps a key to a value, a `DataFrame` maps a column name to a `Series` of column data.
        For example, asking for the `'area'` attribute returns the `Series` object containing the areas we saw earlier:
        """
    )
    return


@app.cell
def _(states):
    states["area"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice the potential point of confusion here: in a two-dimensional NumPy array, `data[0]` will return the first *row*. For a `DataFrame`, `data['col0']` will return the first *column*.
        Because of this, it is probably better to think about ``DataFrame``s as generalized dictionaries rather than generalized arrays, though both ways of looking at the situation can be useful.
        We'll explore more flexible means of indexing ``DataFrame``s in [Data Indexing and Selection](03.02-Data-Indexing-and-Selection.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Constructing DataFrame Objects

        A Pandas `DataFrame` can be constructed in a variety of ways.
        Here we'll explore several examples.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### From a single Series object

        A `DataFrame` is a collection of `Series` objects, and a single-column `DataFrame` can be constructed from a single `Series`:
        """
    )
    return


@app.cell
def _(pd, population):
    pd.DataFrame(population, columns=["population"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### From a list of dicts

        Any list of dictionaries can be made into a `DataFrame`.
        We'll use a simple list comprehension to create some data:
        """
    )
    return


@app.cell
def _(pd):
    data_3 = [{"a": i, "b": 2 * i} for i in range(3)]
    pd.DataFrame(data_3)
    return (data_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Even if some keys in the dictionary are missing, Pandas will fill them in with `NaN` values (i.e., "Not a Number"; see [Handling Missing Data](03.04-Missing-Values.ipynb)):""")
    return


@app.cell
def _(pd):
    pd.DataFrame([{"a": 1, "b": 2}, {"b": 3, "c": 4}])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### From a dictionary of Series objects

        As we saw before, a `DataFrame` can be constructed from a dictionary of `Series` objects as well:
        """
    )
    return


@app.cell
def _(area, pd, population):
    pd.DataFrame({"population": population, "area": area})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### From a two-dimensional NumPy array

        Given a two-dimensional array of data, we can create a `DataFrame` with any specified column and index names.
        If omitted, an integer index will be used for each:
        """
    )
    return


@app.cell
def _(np, pd):
    pd.DataFrame(
        np.random.rand(3, 2), columns=["foo", "bar"], index=["a", "b", "c"]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### From a NumPy structured array

        We covered structured arrays in [Structured Data: NumPy's Structured Arrays](02.09-Structured-Data-NumPy.ipynb).
        A Pandas `DataFrame` operates much like a structured array, and can be created directly from one:
        """
    )
    return


@app.cell
def _(np):
    A = np.zeros(3, dtype=[("A", "i8"), ("B", "f8")])
    A
    return (A,)


@app.cell
def _(A, pd):
    pd.DataFrame(A)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Pandas Index Object

        As you've seen, the `Series` and `DataFrame` objects both contain an explicit *index* that lets you reference and modify data.
        This `Index` object is an interesting structure in itself, and it can be thought of either as an *immutable array* or as an *ordered set* (technically a multiset, as `Index` objects may contain repeated values).
        Those views have some interesting consequences in terms of the operations available on `Index` objects.
        As a simple example, let's construct an `Index` from a list of integers:
        """
    )
    return


@app.cell
def _(pd):
    ind = pd.Index([2, 3, 5, 7, 11])
    ind
    return (ind,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Index as Immutable Array

        The `Index` in many ways operates like an array.
        For example, we can use standard Python indexing notation to retrieve values or slices:
        """
    )
    return


@app.cell
def _(ind):
    ind[1]
    return


@app.cell
def _(ind):
    ind[::2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`Index` objects also have many of the attributes familiar from NumPy arrays:""")
    return


@app.cell
def _(ind):
    print(ind.size, ind.shape, ind.ndim, ind.dtype)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""One difference between `Index` objects and NumPy arrays is that the indices are immutable—that is, they cannot be modified via the normal means:""")
    return


@app.cell
def _(ind):
    ind[1] = 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This immutability makes it safer to share indices between multiple ``DataFrame``s and arrays, without the potential for side effects from inadvertent index modification.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Index as Ordered Set

        Pandas objects are designed to facilitate operations such as joins across datasets, which depend on many aspects of set arithmetic.
        The `Index` object follows many of the conventions used by Python's built-in `set` data structure, so that unions, intersections, differences, and other combinations can be computed in a familiar way:
        """
    )
    return


@app.cell
def _(pd):
    indA = pd.Index([1, 3, 5, 7, 9])
    indB = pd.Index([2, 3, 5, 7, 11])
    return indA, indB


@app.cell
def _(indA, indB):
    indA.intersection(indB)
    return


@app.cell
def _(indA, indB):
    indA.union(indB)
    return


@app.cell
def _(indA, indB):
    indA.symmetric_difference(indB)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

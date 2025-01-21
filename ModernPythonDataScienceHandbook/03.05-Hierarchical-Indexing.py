import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Hierarchical Indexing
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Up to this point we've been focused primarily on one-dimensional and two-dimensional data, stored in Pandas `Series` and `DataFrame` objects, respectively.
        Often it is useful to go beyond this and store higher-dimensional data—that is, data indexed by more than one or two keys.
        Early Pandas versions provided `Panel` and `Panel4D` objects that could be thought of as 3D or 4D analogs to the 2D `DataFrame`, but they were somewhat clunky to use in practice. A far more common pattern for handling higher-dimensional data is to make use of *hierarchical indexing* (also known as *multi-indexing*) to incorporate multiple index *levels* within a single index.
        In this way, higher-dimensional data can be compactly represented within the familiar one-dimensional `Series` and two-dimensional `DataFrame` objects.
        (If you're interested in true *N*-dimensional arrays with Pandas-style flexible indices, you can look into the excellent [Xarray package](https://xarray.pydata.org/).)

        In this chapter, we'll explore the direct creation of `MultiIndex` objects; considerations when indexing, slicing, and computing statistics across multiply indexed data; and useful routines for converting between simple and hierarchically indexed representations of data.

        We begin with the standard imports:
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    return np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A Multiply Indexed Series

        Let's start by considering how we might represent two-dimensional data within a one-dimensional `Series`.
        For concreteness, we will consider a series of data where each point has a character and numerical key.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The Bad Way

        Suppose you would like to track data about states from two different years.
        Using the Pandas tools we've already covered, you might be tempted to simply use Python tuples as keys:
        """
    )
    return


@app.cell
def _(pd):
    index = [('California', 2010), ('California', 2020),
             ('New York', 2010), ('New York', 2020),
             ('Texas', 2010), ('Texas', 2020)]
    populations = [37253956, 39538223,
                   19378102, 20201249,
                   25145561, 29145505]
    pop = pd.Series(populations, index=index)
    pop
    return index, pop, populations


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With this indexing scheme, you can straightforwardly index or slice the series based on this tuple index:
        """
    )
    return


@app.cell
def _(pop):
    pop[('California', 2020):('Texas', 2010)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But the convenience ends there. For example, if you need to select all values from 2010, you'll need to do some messy (and potentially slow) munging to make it happen:
        """
    )
    return


@app.cell
def _(pop):
    pop[[i for i in pop.index if i[1] == 2010]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This produces the desired result, but is not as clean (or as efficient for large datasets) as the slicing syntax we've grown to love in Pandas.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The Better Way: The Pandas MultiIndex
        Fortunately, Pandas provides a better way.
        Our tuple-based indexing is essentially a rudimentary multi-index, and the Pandas `MultiIndex` type gives us the types of operations we wish to have.
        We can create a multi-index from the tuples as follows:
        """
    )
    return


@app.cell
def _(index, pd):
    index_1 = pd.MultiIndex.from_tuples(index)
    return (index_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `MultiIndex` represents multiple *levels* of indexing—in this case, the state names and the years—as well as multiple *labels* for each data point which encode these levels.

        If we reindex our series with this `MultiIndex`, we see the hierarchical representation of the data:
        """
    )
    return


@app.cell
def _(index_1, pop):
    pop_1 = pop.reindex(index_1)
    pop_1
    return (pop_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here the first two columns of the Series representation show the multiple index values, while the third column shows the data.
        Notice that some entries are missing in the first column: in this multi-index representation, any blank entry indicates the same value as the line above it.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now to access all data for which the second index is 2020, we can use the Pandas slicing notation:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1[:, 2020]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The result is a singly indexed Series with just the keys we're interested in.
        This syntax is much more convenient (and the operation is much more efficient!) than the home-spun tuple-based multi-indexing solution that we started with.
        We'll now further discuss this sort of indexing operation on hierarchically indexed data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### MultiIndex as Extra Dimension

        You might notice something else here: we could easily have stored the same data using a simple `DataFrame` with index and column labels.
        In fact, Pandas is built with this equivalence in mind. The `unstack` method will quickly convert a multiply indexed `Series` into a conventionally indexed `DataFrame`:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_df = pop_1.unstack()
    pop_df
    return (pop_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Naturally, the ``stack`` method provides the opposite operation:
        """
    )
    return


@app.cell
def _(pop_df):
    pop_df.stack()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Seeing this, you might wonder why would we would bother with hierarchical indexing at all.
        The reason is simple: just as we were able to use multi-indexing to manipulate two-dimensional data within a one-dimensional `Series`, we can also use it to manipulate data of three or more dimensions in a `Series` or `DataFrame`.
        Each extra level in a multi-index represents an extra dimension of data; taking advantage of this property gives us much more flexibility in the types of data we can represent. Concretely, we might want to add another column of demographic data for each state at each year (say, population under 18); with a `MultiIndex` this is as easy as adding another column to the ``DataFrame``:
        """
    )
    return


@app.cell
def _(pd, pop_1):
    pop_df_1 = pd.DataFrame({'total': pop_1, 'under18': [9284094, 8898092, 4318033, 4181528, 6879014, 7432474]})
    pop_df_1
    return (pop_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In addition, all the ufuncs and other functionality discussed in [Operating on Data in Pandas](03.03-Operations-in-Pandas.ipynb) work with hierarchical indices as well.
        Here we compute the fraction of people under 18 by year, given the above data:
        """
    )
    return


@app.cell
def _(pop_df_1):
    f_u18 = pop_df_1['under18'] / pop_df_1['total']
    f_u18.unstack()
    return (f_u18,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This allows us to easily and quickly manipulate and explore even high-dimensional data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Methods of MultiIndex Creation

        The most straightforward way to construct a multiply indexed `Series` or `DataFrame` is to simply pass a list of two or more index arrays to the constructor. For example:
        """
    )
    return


@app.cell
def _(np, pd):
    df = pd.DataFrame(np.random.rand(4, 2),
                      index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                      columns=['data1', 'data2'])
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The work of creating the ``MultiIndex`` is done in the background.

        Similarly, if you pass a dictionary with appropriate tuples as keys, Pandas will automatically recognize this and use a ``MultiIndex`` by default:
        """
    )
    return


@app.cell
def _(pd):
    data = {('California', 2010): 37253956,
            ('California', 2020): 39538223,
            ('New York', 2010): 19378102,
            ('New York', 2020): 20201249,
            ('Texas', 2010): 25145561,
            ('Texas', 2020): 29145505}
    pd.Series(data)
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Nevertheless, it is sometimes useful to explicitly create a `MultiIndex`; we'll look at a couple of methods for doing this next.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explicit MultiIndex Constructors

        For more flexibility in how the index is constructed, you can instead use the constructor methods available in the `pd.MultiIndex` class.
        For example, as we did before, you can construct a `MultiIndex` from a simple list of arrays giving the index values within each level:
        """
    )
    return


@app.cell
def _(pd):
    pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or you can construct it from a list of tuples giving the multiple index values of each point:
        """
    )
    return


@app.cell
def _(pd):
    pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can even construct it from a Cartesian product of single indices:
        """
    )
    return


@app.cell
def _(pd):
    pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Similarly, you can construct a `MultiIndex` directly using its internal encoding by passing `levels` (a list of lists containing available index values for each level) and `codes` (a list of lists that reference these labels):
        """
    )
    return


@app.cell
def _(pd):
    pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
                  codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Any of these objects can be passed as the `index` argument when creating a `Series` or `DataFrame`, or be passed to the `reindex` method of an existing `Series` or `DataFrame`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### MultiIndex Level Names

        Sometimes it is convenient to name the levels of the `MultiIndex`.
        This can be accomplished by passing the `names` argument to any of the previously discussed `MultiIndex` constructors, or by setting the `names` attribute of the index after the fact:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1.index.names = ['state', 'year']
    pop_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With more involved datasets, this can be a useful way to keep track of the meaning of various index values.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### MultiIndex for Columns

        In a `DataFrame`, the rows and columns are completely symmetric, and just as the rows can have multiple levels of indices, the columns can have multiple levels as well.
        Consider the following, which is a mock-up of some (somewhat realistic) medical data:
        """
    )
    return


@app.cell
def _(np, pd):
    index_2 = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=['year', 'visit'])
    columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']], names=['subject', 'type'])
    data_1 = np.round(np.random.randn(4, 6), 1)
    data_1[:, ::2] = data_1[:, ::2] * 10
    data_1 = data_1 + 37
    health_data = pd.DataFrame(data_1, index=index_2, columns=columns)
    health_data
    return columns, data_1, health_data, index_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is fundamentally four-dimensional data, where the dimensions are the subject, the measurement type, the year, and the visit number.
        With this in place we can, for example, index the top-level column by the person's name and get a full `DataFrame` containing just that person's information:
        """
    )
    return


@app.cell
def _(health_data):
    health_data['Guido']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Indexing and Slicing a MultiIndex

        Indexing and slicing on a `MultiIndex` is designed to be intuitive, and it helps if you think about the indices as added dimensions.
        We'll first look at indexing multiply indexed `Series`, and then multiply indexed `DataFrame` objects.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Multiply Indexed Series

        Consider the multiply indexed `Series` of state populations we saw earlier:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can access single elements by indexing with multiple terms:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1['California', 2010]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `MultiIndex` also supports *partial indexing*, or indexing just one of the levels in the index.
        The result is another `Series`, with the lower-level indices maintained:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1['California']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Partial slicing is available as well, as long as the `MultiIndex` is sorted (see the discussion in [Sorted and Unsorted Indices](#Sorted-and-unsorted-indices)):
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1.loc['California':'New York']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With sorted indices, partial indexing can be performed on lower levels by passing an empty slice in the first index:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1[:, 2010]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Other types of indexing and selection (discussed in [Data Indexing and Selection](03.02-Data-Indexing-and-Selection.ipynb)) work as well; for example, selection based on Boolean masks:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1[pop_1 > 22000000]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Selection based on fancy indexing also works:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1[['California', 'Texas']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Multiply Indexed DataFrames

        A multiply indexed `DataFrame` behaves in a similar manner.
        Consider our toy medical `DataFrame` from before:
        """
    )
    return


@app.cell
def _(health_data):
    health_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Remember that columns are primary in a `DataFrame`, and the syntax used for multiply indexed `Series` applies to the columns.
        For example, we can recover Guido's heart rate data with a simple operation:
        """
    )
    return


@app.cell
def _(health_data):
    health_data['Guido', 'HR']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Also, as with the single-index case, we can use the `loc`, `iloc`, and `ix` indexers introduced in [Data Indexing and Selection](03.02-Data-Indexing-and-Selection.ipynb). For example:
        """
    )
    return


@app.cell
def _(health_data):
    health_data.iloc[:2, :2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These indexers provide an array-like view of the underlying two-dimensional data, but each individual index in `loc` or `iloc` can be passed a tuple of multiple indices. For example:
        """
    )
    return


@app.cell
def _(health_data):
    health_data.loc[:, ('Bob', 'HR')]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Working with slices within these index tuples is not especially convenient; trying to create a slice within a tuple will lead to a syntax error:
        """
    )
    return


app._unparsable_cell(
    r"""
    health_data.loc[(:, 1), (:, 'HR')]
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You could get around this by building the desired slice explicitly using Python's built-in `slice` function, but a better way in this context is to use an `IndexSlice` object, which Pandas provides for precisely this situation.
        For example:
        """
    )
    return


@app.cell
def _(health_data, pd):
    idx = pd.IndexSlice
    health_data.loc[idx[:, 1], idx[:, 'HR']]
    return (idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As you can see, there are many ways to interact with data in multiply indexed `Series` and ``DataFrame``s, and as with many tools in this book the best way to become familiar with them is to try them out!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Rearranging Multi-Indexes

        One of the keys to working with multiply indexed data is knowing how to effectively transform the data.
        There are a number of operations that will preserve all the information in the dataset, but rearrange it for the purposes of various computations.
        We saw a brief example of this in the `stack` and `unstack` methods, but there are many more ways to finely control the rearrangement of data between hierarchical indices and columns, and we'll explore them here.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sorted and Unsorted Indices

        Earlier I briefly mentioned a caveat, but I should emphasize it more here.
        *Many of the `MultiIndex` slicing operations will fail if the index is not sorted.*
        Let's take a closer look.

        We'll start by creating some simple multiply indexed data where the indices are *not lexographically sorted*:
        """
    )
    return


@app.cell
def _(np, pd):
    index_3 = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
    data_2 = pd.Series(np.random.rand(6), index=index_3)
    data_2.index.names = ['char', 'int']
    data_2
    return data_2, index_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we try to take a partial slice of this index, it will result in an error:
        """
    )
    return


@app.cell
def _(data_2):
    try:
        data_2['a':'b']
    except KeyError as e:
        print('KeyError', e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Although it is not entirely clear from the error message, this is the result of the `MultiIndex` not being sorted.
        For various reasons, partial slices and other similar operations require the levels in the `MultiIndex` to be in sorted (i.e., lexographical) order.
        Pandas provides a number of convenience routines to perform this type of sorting, such as the `sort_index` and `sortlevel` methods of the `DataFrame`.
        We'll use the simplest, `sort_index`, here:
        """
    )
    return


@app.cell
def _(data_2):
    data_3 = data_2.sort_index()
    data_3
    return (data_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With the index sorted in this way, partial slicing will work as expected:
        """
    )
    return


@app.cell
def _(data_3):
    data_3['a':'b']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Stacking and Unstacking Indices

        As we saw briefly before, it is possible to convert a dataset from a stacked multi-index to a simple two-dimensional representation, optionally specifying the level to use:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1.unstack(level=0)
    return


@app.cell
def _(pop_1):
    pop_1.unstack(level=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The opposite of `unstack` is `stack`, which here can be used to recover the original series:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_1.unstack().stack()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Index Setting and Resetting

        Another way to rearrange hierarchical data is to turn the index labels into columns; this can be accomplished with the `reset_index` method.
        Calling this on the population dictionary will result in a `DataFrame` with `state` and `year` columns holding the information that was formerly in the index.
        For clarity, we can optionally specify the name of the data for the column representation:
        """
    )
    return


@app.cell
def _(pop_1):
    pop_flat = pop_1.reset_index(name='population')
    pop_flat
    return (pop_flat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A common pattern is to build a `MultiIndex` from the column values.
        This can be done with the `set_index` method of the `DataFrame`, which returns a multiply indexed `DataFrame`:
        """
    )
    return


@app.cell
def _(pop_flat):
    pop_flat.set_index(['state', 'year'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In practice, this type of reindexing is one of the more useful patterns when exploring real-world datasets.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

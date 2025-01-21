import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data Indexing and Selection""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In [Part 2](02.00-Introduction-to-NumPy.ipynb), we looked in detail at methods and tools to access, set, and modify values in NumPy arrays.
        These included indexing (e.g., `arr[2, 1]`), slicing (e.g., `arr[:, 1:5]`), masking (e.g., `arr[arr > 0]`), fancy indexing (e.g., `arr[0, [1, 5]]`), and combinations thereof (e.g., `arr[:, [1, 5]]`).
        Here we'll look at similar means of accessing and modifying values in Pandas `Series` and `DataFrame` objects.
        If you have used the NumPy patterns, the corresponding patterns in Pandas will feel very familiar, though there are a few quirks to be aware of.

        We'll start with the simple case of the one-dimensional `Series` object, and then move on to the more complicated two-dimensional `DataFrame` object.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data Selection in Series

        As you saw in the previous chapter, a `Series` object acts in many ways like a one-dimensional NumPy array, and in many ways like a standard Python dictionary.
        If you keep these two overlapping analogies in mind, it will help you understand the patterns of data indexing and selection in these arrays.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Series as Dictionary

        Like a dictionary, the `Series` object provides a mapping from a collection of keys to a collection of values:
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    data = pd.Series([0.25, 0.5, 0.75, 1.0],
                     index=['a', 'b', 'c', 'd'])
    data
    return data, pd


@app.cell
def _(data):
    data['b']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also use dictionary-like Python expressions and methods to examine the keys/indices and values:""")
    return


@app.cell
def _(data):
    'a' in data
    return


@app.cell
def _(data):
    data.keys()
    return


@app.cell
def _(data):
    list(data.items())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        `Series` objects can also be modified with a dictionary-like syntax.
        Just as you can extend a dictionary by assigning to a new key, you can extend a `Series` by assigning to a new index value:
        """
    )
    return


@app.cell
def _(data):
    data['e'] = 1.25
    data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This easy mutability of the objects is a convenient feature: under the hood, Pandas is making decisions about memory layout and data copying that might need to take place, and the user generally does not need to worry about these issues.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Series as One-Dimensional Array""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A `Series` builds on this dictionary-like interface and provides array-style item selection via the same basic mechanisms as NumPy arrays—that is, slices, masking, and fancy indexing.
        Examples of these are as follows:
        """
    )
    return


@app.cell
def _(data):
    # slicing by explicit index
    data['a':'c']
    return


@app.cell
def _(data):
    # slicing by implicit integer index
    data[0:2]
    return


@app.cell
def _(data):
    # masking
    data[(data > 0.3) & (data < 0.8)]
    return


@app.cell
def _(data):
    # fancy indexing
    data[['a', 'e']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Of these, slicing may be the source of the most confusion.
        Notice that when slicing with an explicit index (e.g., `data['a':'c']`), the final index is *included* in the slice, while when slicing with an implicit index (e.g., `data[0:2]`), the final index is *excluded* from the slice.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Indexers: loc and iloc

        If your `Series` has an explicit integer index, an indexing operation such as `data[1]` will use the explicit indices, while a slicing operation like `data[1:3]` will use the implicit Python-style indices:
        """
    )
    return


@app.cell
def _(pd):
    data_1 = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
    data_1
    return (data_1,)


@app.cell
def _(data_1):
    data_1[1]
    return


@app.cell
def _(data_1):
    data_1[1:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because of this potential confusion in the case of integer indexes, Pandas provides some special *indexer* attributes that explicitly expose certain indexing schemes.
        These are not functional methods, but attributes that expose a particular slicing interface to the data in the `Series`.

        First, the `loc` attribute allows indexing and slicing that always references the explicit index:
        """
    )
    return


@app.cell
def _(data_1):
    data_1.loc[1]
    return


@app.cell
def _(data_1):
    data_1.loc[1:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `iloc` attribute allows indexing and slicing that always references the implicit Python-style index:""")
    return


@app.cell
def _(data_1):
    data_1.iloc[1]
    return


@app.cell
def _(data_1):
    data_1.iloc[1:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One guiding principle of Python code is that "explicit is better than implicit."
        The explicit nature of `loc` and `iloc` makes them helpful in maintaining clean and readable code; especially in the case of integer indexes, using them consistently can prevent subtle bugs due to the mixed indexing/slicing convention.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data Selection in DataFrames

        Recall that a `DataFrame` acts in many ways like a two-dimensional or structured array, and in other ways like a dictionary of `Series` structures sharing the same index.
        These analogies can be helpful to keep in mind as we explore data selection within this structure.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### DataFrame as Dictionary

        The first analogy we will consider is the `DataFrame` as a dictionary of related `Series` objects.
        Let's return to our example of areas and populations of states:
        """
    )
    return


@app.cell
def _(pd):
    area = pd.Series({'California': 423967, 'Texas': 695662, 'Florida': 170312, 'New York': 141297, 'Pennsylvania': 119280})
    pop = pd.Series({'California': 39538223, 'Texas': 29145505, 'Florida': 21538187, 'New York': 20201249, 'Pennsylvania': 13002700})
    data_2 = pd.DataFrame({'area': area, 'pop': pop})
    data_2
    return area, data_2, pop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The individual `Series` that make up the columns of the `DataFrame` can be accessed via dictionary-style indexing of the column name:""")
    return


@app.cell
def _(data_2):
    data_2['area']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Equivalently, we can use attribute-style access with column names that are strings:""")
    return


@app.cell
def _(data_2):
    data_2.area
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Though this is a useful shorthand, keep in mind that it does not work for all cases!
        For example, if the column names are not strings, or if the column names conflict with methods of the `DataFrame`, this attribute-style access is not possible.
        For example, the `DataFrame` has a `pop` method, so `data.pop` will point to this rather than the `pop` column:
        """
    )
    return


@app.cell
def _(data_2):
    data_2.pop is data_2['pop']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In particular, you should avoid the temptation to try column assignment via attributes (i.e., use `data['pop'] = z` rather than `data.pop = z`).

        Like with the `Series` objects discussed earlier, this dictionary-style syntax can also be used to modify the object, in this case adding a new column:
        """
    )
    return


@app.cell
def _(data_2):
    data_2['density'] = data_2['pop'] / data_2['area']
    data_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This shows a preview of the straightforward syntax of element-by-element arithmetic between `Series` objects; we'll dig into this further in [Operating on Data in Pandas](03.03-Operations-in-Pandas.ipynb).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### DataFrame as Two-Dimensional Array

        As mentioned previously, we can also view the `DataFrame` as an enhanced two-dimensional array.
        We can examine the raw underlying data array using the `values` attribute:
        """
    )
    return


@app.cell
def _(data_2):
    data_2.values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With this picture in mind, many familiar array-like operations can be done on the `DataFrame` itself.
        For example, we can transpose the full `DataFrame` to swap rows and columns:
        """
    )
    return


@app.cell
def _(data_2):
    data_2.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When it comes to indexing of a `DataFrame` object, however, it is clear that the dictionary-style indexing of columns precludes our ability to simply treat it as a NumPy array.
        In particular, passing a single index to an array accesses a row:
        """
    )
    return


@app.cell
def _(data_2):
    data_2.values[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""and passing a single "index" to a `DataFrame` accesses a column:""")
    return


@app.cell
def _(data_2):
    data_2['area']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Thus, for array-style indexing, we need another convention.
        Here Pandas again uses the `loc` and `iloc` indexers mentioned earlier.
        Using the `iloc` indexer, we can index the underlying array as if it were a simple NumPy array (using the implicit Python-style index), but the `DataFrame` index and column labels are maintained in the result:
        """
    )
    return


@app.cell
def _(data_2):
    data_2.iloc[:3, :2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Similarly, using the `loc` indexer we can index the underlying data in an array-like style but using the explicit index and column names:""")
    return


@app.cell
def _(data_2):
    data_2.loc[:'Florida', :'pop']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Any of the familiar NumPy-style data access patterns can be used within these indexers.
        For example, in the `loc` indexer we can combine masking and fancy indexing as follows:
        """
    )
    return


@app.cell
def _(data_2):
    data_2.loc[data_2.density > 120, ['pop', 'density']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Any of these indexing conventions may also be used to set or modify values; this is done in the standard way that you might be accustomed to from working with NumPy:""")
    return


@app.cell
def _(data_2):
    data_2.iloc[0, 2] = 90
    data_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To build up your fluency in Pandas data manipulation, I suggest spending some time with a simple `DataFrame` and exploring the types of indexing, slicing, masking, and fancy indexing that are allowed by these various indexing approaches.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Additional Indexing Conventions

        There are a couple of extra indexing conventions that might seem at odds with the preceding discussion, but nevertheless can be useful in practice.
        First, while *indexing* refers to columns, *slicing* refers to rows:
        """
    )
    return


@app.cell
def _(data_2):
    data_2['Florida':'New York']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Such slices can also refer to rows by number rather than by index:""")
    return


@app.cell
def _(data_2):
    data_2[1:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Similarly, direct masking operations are interpreted row-wise rather than column-wise:""")
    return


@app.cell
def _(data_2):
    data_2[data_2.density > 120]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""These two conventions are syntactically similar to those on a NumPy array, and while they may not precisely fit the mold of the Pandas conventions, they are included due to their practical utility.""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.2.5",
#     "pandas==2.2.3",
#     "polars==1.29.0",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import pandas as pd
    return mo, np, pd, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # DataFrames
    Author: Raine Hoang

    In this tutorial, we will go over the central data structure for structured data, DataFrames. There are a multitude of packages that work with DataFrames, but we will be focusing on the way Polars uses them the different options it provides.

    /// Note
    The following tutorial has been adapted from the Polars [documentation](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Defining a DataFrame

    At the most basic level, all that you need to do in order to create a DataFrame in Polars is to use the .DataFrame() method and pass in some data into the data parameter. However, there are restrictions as to what exactly you can pass into this method.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### What Can Be a DataFrame?""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    There are [5 data types](https://github.com/pola-rs/polars/blob/py-1.29.0/py-polars/polars/dataframe/frame.py#L197) that can be converted into a DataFrame.

    1. Dictionary
    2. Sequence
    3. NumPy Array
    4. Series
    5. Pandas DataFrame
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Dictionary

    Dictionaries are structures that store data as `key:value` pairs. Let's say we have the following dictionary:
    """
    )
    return


@app.cell
def _():
    dct_data = {"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"], "col3": [1.2, 4.2, 6.4, 3.7]}
    dct_data
    return (dct_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In order to convert this dictionary into a DataFrame, we simply need to pass it into the data parameter in the `.DataFrame()` method like so.""")
    return


@app.cell
def _(dct_data, pl):
    dct_df = pl.DataFrame(data = dct_data)
    dct_df
    return (dct_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this case, Polars turned each of the lists in the dictionary into a column in the DataFrame. 

    The other data structures will follow a similar pattern when converting them to DataFrames.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### Sequence

    Sequences are data structures that contain collections of items, which can be accessed using its index. Examples of sequences are lists, tuples, and strings. We will be using a list of lists in order to demonstrate how to convert a sequence in a DataFrame.
    """
    )
    return


@app.cell
def _():
    seq_data = [[1, 2, 3, 4], ["a", "b", "c", "d"], [1.2, 4.2, 6.4, 3.7]]
    seq_data
    return (seq_data,)


@app.cell
def _(pl, seq_data):
    seq_df = pl.DataFrame(data = seq_data)
    seq_df
    return (seq_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Notice that since we didn't specify the column names, Polars automatically named them `column_0`, `column_1`, and `column_2`. Later, we will show you how to specify the names of the columns.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### NumPy Array

    NumPy arrays are considered a sequence of items that can also be accessed using its index. An important thing to note is that all of the items in an array must have the same data type.
    """
    )
    return


@app.cell
def _(np):
    arr_data = np.array([np.array([1, 2, 3, 4]), np.array(["a", "b", "c", "d"]), np.array([1.2, 4.2, 6.4, 3.7])])
    arr_data
    return (arr_data,)


@app.cell
def _(arr_data, pl):
    arr_df = pl.DataFrame(data = arr_data)
    arr_df
    return (arr_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Notice that each inner array is a row in the DataFrame, not a column like the previous methods discussed. Later, we will go over how to tell Polars if we the information in the data structure to be presented as rows or columns.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### Series

    Series are a way to store a single column in a DataFrame and all entries in a series must have the same data type. You can combine these series together to form one DataFrame.
    """
    )
    return


@app.cell
def _(pl):
    pl_series = [pl.Series([1, 2, 3, 4]), pl.Series(["a", "b", "c", "d"]), pl.Series([1.2, 4.2, 6.4, 3.7])]
    pl_series
    return (pl_series,)


@app.cell
def _(pl, pl_series):
    series_df = pl.DataFrame(data = pl_series)
    series_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### Pandas DataFrame

    Another popular package that utilizes DataFrames is pandas. By passing in a pandas DataFrame into .DataFrame(), you can easily convert it into a Polars DataFrame.
    """
    )
    return


@app.cell
def _(dct_data, pd):
    # Creates a DataFrame from a dictionary using pandas package
    pd_df = pd.DataFrame(data = dct_data)

    pd_df
    return (pd_df,)


@app.cell
def _(pd_df, pl):
    # Takes pandas DataFrame and converts it into Polars DataFrame
    pl_df = pl.DataFrame(data = pd_df)

    pl_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now that we've looked over what can be converted into a DataFrame and the basics of it, let's look at the structure of the DataFrame.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## DataFrame Structure

    Let's recall one of the DataFrames we defined earlier.
    """
    )
    return


@app.cell
def _(dct_df):
    dct_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can see that this DataFrame has 4 rows and 3 columns as indicated by the text beneath the DataFrame. Each column has a name that can be used to access the data within that column. In this case, the names are: "col1", "col2", and "col3". Below the column name, there is text that indicates the data type stored within that column. "col1" has the text "i64" underneath its name, meaning that that column stores integers. "col2" stores strings as seen by the "str" under the column name. Finally, "col3" stores floats as it has "f64" under the column name. Polars will automatically assume the data types stored in each column, but we will go over a way to specify it later in this tutorial. Each column can only hold one data type at a time, so you can't have a string and an integer in the same column.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Parameters

    On top of the "data" parameter, there are 6 additional parameters you can specify:

    1. schema
    2. schema_overrides
    3. strict
    4. orient
    5. infer_schema_length
    6. nan_to_null
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Schema

    Let's recall the DataFrame we created using a sequence.
    """
    )
    return


@app.cell
def _(seq_df):
    seq_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can see that the column names and data type were inferred by Polars. The schema parameter allows us to specify the column names and data type we want for each column. There are 3 ways you can use this parameter. The first way involves using a dictionary to define the following key value pair: column name:data type.""")
    return


@app.cell
def _(pl, seq_data):
    pl.DataFrame(seq_data, schema = {"integers": pl.Int16, "strings": pl.String, "floats": pl.Float32})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can also do this using a list of (column name, data type) pairs instead of a dictionary.""")
    return


@app.cell
def _(pl, seq_data):
    pl.DataFrame(seq_data, schema = [("integers", pl.Int16), ("strings", pl.String), ("floats", pl.Float32)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Notice how both the column names and the data type (text underneath the column name) is different from the original `seq_df`. If you only wanted to specify the column names and let Polars assume the data type, you can do so using a list of column names.""")
    return


@app.cell
def _(pl, seq_data):
    pl.DataFrame(seq_data, schema = ["integers", "strings", "floats"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The text under the column names is different from the previous two DataFrames we created since we didn't explicitly tell Polars what data type we wanted in each column.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Schema_Overrides

    If you only wanted to specify the data type of specific columns and let Polars infer the rest, you can use the schema_overrides parameter for that. This parameter requires that you pass in a dictionary where the key value pair is column name:data type. Unlike the schema parameter, the column name must match the name already present in the DataFrame as that is how Polars will identify which column you want to specify the data type. If you use a column name that doesn't already exist, Polars won't be able to change the data type.
    """
    )
    return


@app.cell
def _(pl, seq_data):
    pl.DataFrame(seq_data, schema_overrides = {"column_0": pl.Int16})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Notice here that only the data type in the first column changed while Polars inferred the rest.

    It is important to note that if you only use the schema_overrides parameter, you are limited to how much you can change the data type. In the example above, we were able to change the data type from int32 to int16 without any further parameters since the data type is still an integer. However, if we wanted to change the first column to be a string, we would get an error as Polars has already strictly set the schema to only take in integer values.
    """
    )
    return


@app.cell
def _(pl, seq_data):
    pl.DataFrame(seq_data, schema_overrides = {"column_0": pl.String})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If we wanted to use schema_override to completely change the data type of the column, we need an additional parameter: strict.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Strict

    The strict parameter allows you to specify if you want a column's data type to be enforced with flexibility or not. When set to `True`, Polars will raise an error if there is a data type that doesn't match the data type the column is expecting. It will not attempt to type cast it to the correct data type as Polars prioritizes that all the data can be converted without any loss or error. When set to `False`, Polars will attempt to type cast the data into the data type the column wants. If it is unable to successfully convert the data type, the value will be replaced with a null value.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's see an example of what happens when strict is set to `True`. The cell below should show an error.""")
    return


@app.cell
def _(pl):
    data = [[1, "a", 2]]

    pl.DataFrame(data = data, strict = True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now let's try setting strict to `False`.""")
    return


@app.cell
def _(pl, seq_data):
    pl.DataFrame(seq_data, schema_overrides = {"column_0": pl.String}, strict = False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Since we allowed for Polars to change the schema by setting strict to `False`, we were able to cast the first column to be strings.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    #### Orient

    Let's recall the DataFrame we made by using an array and the data used to make it.
    """
    )
    return


@app.cell
def _(arr_data):
    arr_data
    return


@app.cell
def _(arr_df):
    arr_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Notice how Polars decided to make each inner array a row in the DataFrame. If we wanted to make it so that each inner array was a column instead of a row, all we would need to do is pass `"col"` into the orient parameter.""")
    return


@app.cell
def _(arr_data, pl):
    pl.DataFrame(data = arr_data, orient = "col")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If we wanted to do the opposite, then we pass `"row"` into the orient parameter.""")
    return


@app.cell
def _(seq_df):
    seq_df
    return


@app.cell
def _(pl, seq_data):
    pl.DataFrame(data = seq_data, orient = "row")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Infer_Schema_Length

    Without setting the schema ourselves, Polars uses the data provided to infer the data types of the columns. It does this by looking at each of the rows in the data provided. You can specify to Polars how many rows to look at by using the infer_schema_length parameter. For example, if you were to set this parameter to 5, then Polars would use the first 5 rows to infer the schema.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### NaN_To_Null

    If there are np.nan values in the data, you can convert them to null values by setting the nan_to_null parameter to `True`.
    """
    )
    return


if __name__ == "__main__":
    app.run()

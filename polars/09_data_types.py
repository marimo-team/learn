# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars==1.18.0",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Types

    Author: [Deb Debnath](https://github.com/debajyotid2)

    **Note**: The following tutorial has been adapted from the Polars [documentation](https://docs.pola.rs/user-guide/concepts/data-types-and-structures/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Polars supports a variety of data types that fall broadly under the following categories:

    - Numeric data types: integers and floating point numbers.
    - Nested data types: lists, structs, and arrays.
    - Temporal: dates, datetimes, times, and time deltas.
    - Miscellaneous: strings, binary data, Booleans, categoricals, enums, and objects.

    All types support missing values represented by `null` which is different from `NaN` used in floating point data types. The numeric datatypes in Polars loosely follow the type system of the Rust language, since its core functionalities are built in Rust.

    [Here](https://docs.pola.rs/api/python/stable/reference/datatypes.html) is a full list of all data types Polars supports.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Series

    A series is a 1-dimensional data structure that can hold only one data type.
    """)
    return


@app.cell
def _(pl):
    s = pl.Series("emojis", ["😀", "🤣", "🥶", "💀", "🤖"])
    s
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unless specified, Polars infers the datatype from the supplied values.
    """)
    return


@app.cell
def _(pl):
    s1 = pl.Series("friends", ["Евгений", "अभिषेक", "秀良", "Federico", "Bob"])
    s2 = pl.Series("uints", [0x00, 0x01, 0x10, 0x11], dtype=pl.UInt8)
    s1.dtype, s2.dtype
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataframe

    A dataframe is a 2-dimensional data structure that contains uniquely named series and can hold multiple data types. Dataframes are more commonly used for data manipulation using the functionality of Polars.

    The snippet below shows how to create a dataframe from a dictionary of lists:
    """)
    return


@app.cell
def _(pl):
    data = pl.DataFrame(
        {
            "Product ID": ["L51172", "M22586", "L51639", "L50250", "M20109"],
            "Type": ["L", "M", "L", "L", "M"],
            "Air temperature": [302.3, 300.8, 302.6, 300, 303.4],  # (K)
            "Machine Failure": [False, True, False, False, True]
        }
    )
    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspecting a dataframe

    Polars has various functions to explore the data in a dataframe. We will use the dataframe `data` defined above in our examples. Alongside we can also see a view of the dataframe rendered by `marimo` as the cells are executed.

    ///note
    We can also use `marimo`'s built in data-inspection elements/features such as  [`mo.ui.dataframe`](https://docs.marimo.io/api/inputs/dataframe/#marimo.ui.dataframe) & [`mo.ui.data_explorer`](https://docs.marimo.io/api/inputs/data_explorer/). For more check out our Polars tutorials at [`marimo learn`](https://marimo-team.github.io/learn/)!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Head

    The function `head` shows the first rows of a dataframe. Unless specified, it shows the first 5 rows.
    """)
    return


@app.cell
def _(data):
    data.head(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Glimpse

    The function `glimpse` is an alternative to `head` to view the first few columns, but displays each line of the output corresponding to a single column. That way, it makes inspecting wider dataframes easier.
    """)
    return


@app.cell
def _(data):
    print(data.glimpse(return_as_string=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Tail

    The `tail` function, just like its name suggests, shows the last rows of a dataframe. Unless the number of rows is specified, it will show the last 5 rows.
    """)
    return


@app.cell
def _(data):
    data.tail(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Sample

    `sample` can be used to show a specified number of randomly selected rows from the dataframe. Unless the number of rows is specified, it will show a single row. `sample` does not preserve order of the rows.
    """)
    return


@app.cell
def _(data):
    import random

    random.seed(42)  # For reproducibility.

    data.sample(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Describe

    The function `describe` describes the summary statistics for all columns of a dataframe.
    """)
    return


@app.cell
def _(data):
    data.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Schema

    A schema is a mapping showing the datatype corresponding to every column of a dataframe. The schema of a dataframe can be viewed using the attribute `schema`.
    """)
    return


@app.cell
def _(data):
    data.schema
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since a schema is a mapping, it can be specified in the form of a Python dictionary. Then this dictionary can be used to specify the schema of a dataframe on definition. If not specified or the entry is `None`, Polars infers the datatype from the contents of the column. Note that if the schema is not specified, it will be inferred automatically by default.
    """)
    return


@app.cell
def _(pl):
    pl.DataFrame(
        {
            "Product ID": ["L51172", "M22586", "L51639", "L50250", "M20109"],
            "Type": ["L", "M", "L", "L", "M"],
            "Air temperature": [302.3, 300.8, 302.6, 300, 303.4],  # (K)
            "Machine Failure": [False, True, False, False, True]
        },
        schema={"Product ID": pl.String, "Type": pl.String, "Air temperature": None, "Machine Failure": None},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sometimes the automatically inferred schema is enough for some columns, but we might wish to override the inference of only some columns. We can specify the schema for those columns using `schema_overrides`.
    """)
    return


@app.cell
def _(pl):
    pl.DataFrame(
        {
            "Product ID": ["L51172", "M22586", "L51639", "L50250", "M20109"],
            "Type": ["L", "M", "L", "L", "M"],
            "Air temperature": [302.3, 300.8, 302.6, 300, 303.4],  # (K)
            "Machine Failure": [False, True, False, False, True]
        },
        schema_overrides={"Air temperature": pl.Float32},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### References

    1. Polars documentation ([link](https://docs.pola.rs/api/python/stable/reference/datatypes.html))
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


if __name__ == "__main__":
    app.run()

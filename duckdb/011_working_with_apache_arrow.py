# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "duckdb==1.2.1",
#     "pyarrow==19.0.1",
#     "polars[pyarrow]==1.25.2",
#     "pandas==2.2.3",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Working with Apache Arrow
    *By [Thomas Liang](https://github.com/thliang01)*
    #
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [Apache Arrow](https://arrow.apache.org/) is a multi-language toolbox for building high performance applications that process and transport large data sets. It is designed to both improve the performance of analytical algorithms and the efficiency of moving data from one system or programming language to another.

        A critical component of Apache Arrow is its in-memory columnar format, a standardized, language-agnostic specification for representing structured, table-like datasets in-memory. This data format has a rich data type system (included nested and user-defined data types) designed to support the needs of analytic database systems, data frame libraries, and more.

        DuckDB has native support for Apache Arrow, which is an in-memory columnar data format. This allows for efficient data transfer between DuckDB and other Arrow-compatible systems, such as Polars and Pandas (via PyArrow).

        In this notebook, we'll explore how to:

        - Create an Arrow table from a DuckDB query.
        - Load an Arrow table into DuckDB.
        - Convert between DuckDB, Arrow, and Polars/Pandas DataFrames.
        """
    )
    return


@app.cell
def _(mo):
    mo.sql(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER,
            name VARCHAR,
            age INTEGER,
            city VARCHAR
        );

        INSERT INTO users VALUES
            (1, 'Alice', 30, 'New York'),
            (2, 'Bob', 24, 'London'),
            (3, 'Charlie', 35, 'Paris'),
            (4, 'David', 29, 'New York'),
            (5, 'Eve', 40, 'London');
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Creating an Arrow Table from a DuckDB Query

        You can directly fetch the results of a DuckDB query as an Apache Arrow table using the `.arrow()` method on the query result.
        """
    )
    return


@app.cell
def _(mo):
    users_arrow_table = mo.sql(  # type: ignore
        """
        SELECT * FROM users WHERE age > 30;
        """
    ).to_arrow()
    return (users_arrow_table,)


@app.cell
def _(users_arrow_table):
    users_arrow_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"The `.arrow()` method returns a `pyarrow.Table` object. We can inspect its schema:")
    return


@app.cell
def _(users_arrow_table):
    users_arrow_table.schema
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Loading an Arrow Table into DuckDB

        You can also register an existing Arrow table (or a Polars/Pandas DataFrame, which uses Arrow under the hood) directly with DuckDB. This allows you to query the in-memory data without any copying, which is highly efficient.
        """
    )
    return


@app.cell
def _(pa):
    # Create an Arrow table in Python
    new_data = pa.table({
        'id': [6, 7],
        'name': ['Fiona', 'George'],
        'age': [22, 45],
        'city': ['Berlin', 'Tokyo']
    })
    return (new_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, we can query this Arrow table `new_data` directly from SQL by embedding it in the query.
        """
    )
    return


@app.cell
def _(mo, new_data):
    mo.sql(
        f"""
        SELECT name, age, city
        FROM new_data
        WHERE age > 30;
        """
    )
    return

# Working in Interoperability with Polars and Pandas

# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(
#         r"""
#         ## 3. Interoperability with Polars and Pandas

#         The real power of DuckDB's Arrow integration comes from its seamless interoperability with data frame libraries like Polars and Pandas. Because they all share the Arrow in-memory format, conversions are often zero-copy and extremely fast.
#         """
#     )
#     return


# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(r"### From DuckDB to Polars/Pandas")
#     return


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    return mo, px


@app.cell
def _():
    import pyarrow as pa
    import polars as pl
    import pandas as pd
    return


if __name__ == "__main__":
    app.run()
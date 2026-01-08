# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb==1.4.3",
#     "kagglehub==0.3.13",
#     "polars==1.36.1",
#     "pyarrow==22.0.0",
#     "sqlalchemy==2.0.45",
#     "sqlglot==28.3.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SQL Features in Marimo and Polars

    _By [etrotta](https://github.com/etrotta)_

    For this Notebook, we'll be using a [hotel booking analytics](https://www.kaggle.com/datasets/alperenmyung/international-hotel-booking-analytics) dataset.

    We will see many ways in which you can use SQL inside of marimo and how each feature interacts with polars, including:
    - How to read data from a SQLite file (or any Database connection)
    - What are SQL Cells in Marimo
    - How to load an SQL query into a DataFrame
    - How to query DataFrames using SQL
    """)
    return


@app.cell
def _(mo, sqlite_engine):
    _df = mo.sql(
        f"""
        SELECT * FROM reviews LIMIT 100
        """,
        engine=sqlite_engine,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will start by using `kagglehub` to download a `.sqlite` file, then create an `SQLAlchemy` engine to let marimo know about the database.
    """)
    return


@app.cell
def _(kagglehub):
    dataset_id = "alperenmyung/international-hotel-booking-analytics"
    cached_file = kagglehub.dataset_download(dataset_id, "booking_db.sqlite")
    return (cached_file,)


@app.cell
def _(cached_file):
    cached_file
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using Marimo's SQL Cells
    """)
    return


@app.cell
def _(cached_file, sqlalchemy):
    sqlite_engine = sqlalchemy.create_engine("sqlite:///" + cached_file)
    return (sqlite_engine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After creating the Engine, you should be able to see it in the **Data Sources** panel in the sidebar. Whenever you create an SQLAlchemy engine as a global variable, Marimo picks up on it and makes it available for use in SQL Cells

    You can use it to consult all tables and their columns, as well as click "Add table to notebook" to get the code to use it, creating our first SQL Cell:
    """)
    return


@app.cell
def _(mo, sqlite_engine):
    _df = mo.sql(
        f"""
        SELECT * FROM hotels LIMIT 10
        """,
        engine=sqlite_engine,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `Output variable:` can be used to save the output as a polars DataFrame you can access later

    For example, fetching all scores then performing a group by in polars
    """)
    return


@app.cell
def _(mo, sqlite_engine):
    polars_age_groups = mo.sql(
        f"""
        SELECT reviews.*, age_group FROM reviews JOIN users ON reviews.user_id = users.user_id LIMIT 1000
        """,
        engine=sqlite_engine,
    )
    return (polars_age_groups,)


@app.cell
def _(pl, polars_age_groups):
    _mean_scores = pl.col("^score_.*$").mean()
    _age_group_start = pl.col("age_group").str.slice(0, 2).cast(int)
    polars_age_groups.group_by("age_group").agg(_mean_scores).sort(_age_group_start)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Although you could also calculate it directly in SQL, this gives you the flexibility to use polars for operations that are harder to describe in SQL
    """)
    return


@app.cell
def _(mo, sqlite_engine):
    _df = mo.sql(
        f"""
        SELECT age_group, AVG(reviews.score_overall) FROM reviews JOIN users ON reviews.user_id = users.user_id GROUP BY age_group
        """,
        engine=sqlite_engine,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also use SQL Cells to query DataFrames via DuckDB, but remember to change the Engine from the SQLite engine into the DuckDB Memory engine when doing so
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        SELECT * FROM polars_age_groups LIMIT 10
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using Polars directly
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Polars also offers some methods to interact with databases and query DataFrames using SQL directly, which you can use inside or outside of marimo the same.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reading data from Databases:
    """)
    return


@app.cell
def _(pl, sqlite_engine):
    hotels = pl.read_database("SELECT * FROM hotels LIMIT 10", sqlite_engine)
    hotels
    return (hotels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Querying DataFrames with SQL:
    """)
    return


@app.cell
def _(hotels):
    hotels.sql("SELECT * from self ORDER BY cleanliness_base DESC LIMIT 5")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using DuckDB
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While marimo's SQL Cells are very practical and polars's direct methods are about as portable as it gets using polars, you can also use other libraries that integrate with polars via Arrow tables or input plugins.

    One example of such integrations is DuckDB, which can be used with polars's Lazy mode as of 1.4.0
    """)
    return


@app.cell
def _(cached_file, duckdb):
    duckdb_conn = duckdb.connect(cached_file)
    return (duckdb_conn,)


@app.cell
def _(duckdb_conn):
    # Loading into a normal DataFrame:
    duckdb_conn.sql("SELECT * FROM hotels LIMIT 10").pl()
    return


@app.cell
def _(duckdb_conn):
    # Loading into a LazyFrame:
    duckdb_conn.sql("SELECT * FROM hotels").pl(lazy=True).limit(10).collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that this is very similar to SQL cells backed by DuckDB, with the biggest difference being that you can control how the result is consumed as opposed to it always being loaded into memory.

    Many features such as querying from DataFrames work the same using DuckDB directly as they do in DuckDB-backed SQL Cells, and vice-versa
    """)
    return


@app.cell
def _(duckdb):
    duckdb.sql("SELECT * FROM hotels").pl(lazy=True).sort("cleanliness_base", descending=True).limit(5).collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Utilities
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    delete_file_button = mo.ui.run_button(label="Delete cached file", kind="warn")
    mo.vstack([mo.md("If you want to delete the downloaded file from your cache"), delete_file_button])
    return (delete_file_button,)


@app.cell
def _(cached_file, delete_file_button, pathlib):
    if delete_file_button.value:
        pathlib.Path(cached_file).unlink()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _():
    import duckdb
    return (duckdb,)


@app.cell
def _():
    import sqlalchemy
    return (sqlalchemy,)


@app.cell
def _():
    import kagglehub
    return (kagglehub,)


@app.cell
def _():
    import pathlib
    return (pathlib,)


if __name__ == "__main__":
    app.run()

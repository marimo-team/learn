# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "duckdb==1.3.2",
#     "pyarrow==19.0.1",
#     "plotly.express",
#     "sqlglot==27.0.0",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Loading Parquet files with DuckDB
    *By [Thomas Liang](https://github.com/thliang01)*
    #
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [Apache Parquet](https://parquet.apache.org/) is a popular columnar storage format, optimized for analytics. Its columnar nature allows query engines like DuckDB to read only the necessary columns, leading to significant performance gains, especially for wide tables.

        DuckDB has excellent, built-in support for reading Parquet files, making it incredibly easy to query and analyze Parquet data directly without a separate loading step.

        In this notebook, we'll explore how to load and analyze Airbnb's stock price data from a remote Parquet file:
        <ul>
            <li>Querying a remote Parquet file directly.</li>
            <li>Using the `read_parquet` function for more control.</li>
            <li>Creating a persistent table from a Parquet file.</li>
            <li>Performing basic data analysis and visualization.</li>
        </ul>
        """
    )
    return


@app.cell
def _():
    AIRBNB_URL = 'https://huggingface.co/datasets/BatteRaquette58/airbnb-stock-price/resolve/main/data/airbnb-stock.parquet'
    return (AIRBNB_URL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Using `FROM` to query Parquet files""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The simplest way to query a Parquet file is to use it directly in a `FROM` clause, just like you would with a table. DuckDB will automatically detect that it's a Parquet file and read it accordingly.

        Let's query a dataset of Airbnb's stock price from Hugging Face.
        """
    )
    return


@app.cell
def _(AIRBNB_URL, mo, null):
    mo.sql(
        f"""
        SELECT *
        FROM '{AIRBNB_URL}'
        LIMIT 5;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Using `read_parquet`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more control, you can use the `read_parquet` table function. This is useful when you need to specify options, for example, when dealing with multiple files or specific data types.
        Some useful options for `read_parquet` include:

        - `binary_as_string=True`: Reads `BINARY` columns as `VARCHAR`.
        - `filename=True`: Adds a `filename` column with the path of the file for each row.
        - `hive_partitioning=True`: Enables reading of Hive-partitioned datasets.

        Here, we'll use `read_parquet` to select only a few relevant columns. This is much more efficient than `SELECT *` because DuckDB only needs to read the data for the columns we specify.
        """
    )
    return


@app.cell
def _(AIRBNB_URL, mo):
    mo.sql(
        f"""
        SELECT Date, Open, "close_last", High, Low
        FROM read_parquet('{AIRBNB_URL}')
        LIMIT 5;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can also read multiple Parquet files at once using a glob pattern. For example, to read all Parquet files in a directory `data/`:

        ```sql
        SELECT * FROM read_parquet('data/*.parquet');
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Creating a table from a Parquet file""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        While querying Parquet files directly is powerful, sometimes it's useful to load the data into a persistent table within your DuckDB database. This can simplify subsequent queries and is a good practice if you'll be accessing the data frequently.
        """
    )
    return


@app.cell
def _(AIRBNB_URL, mo):
    stock_table = mo.sql(
        f"""
        CREATE OR REPLACE TABLE airbnb_stock AS
        SELECT * FROM read_parquet('{AIRBNB_URL}');
        """
    )
    return airbnb_stock, stock_table


@app.cell(hide_code=True)
def _(mo, stock_table):
    mo.md(
        f"""
    {stock_table}

    Now that the `airbnb_stock` table is created, we can query it like any other SQL table.
    """
    )
    return


@app.cell
def _(airbnb_stock, mo):
    mo.sql(
        f"""
        SELECT * FROM airbnb_stock LIMIT 5;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analysis and Visualization""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's perform a simple analysis: plotting the closing stock price over time.""")
    return


@app.cell
def _(airbnb_stock, mo):
    stock_data = mo.sql(
        f"""
        SELECT
        CAST(to_timestamp(Date) AS DATE) AS "Date",
            "close_last"
        FROM airbnb_stock
        ORDER BY "Date";
        """
    )
    return (stock_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we can easily visualize this result using marimo's integration with plotting libraries like Plotly.""")
    return


@app.cell
def _(px, stock_data):
    px.line(
        stock_data,
        x="Date",
        y="close_last",
        title="Airbnb (ABNB) Stock Price Over Time",
        labels={"Date": "Date", "close_last": "Closing Price (USD)"},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Conclusion""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this notebook, we've seen how easy it is to work with Parquet files in DuckDB. We learned how to:
    <ul>
        <li>Query Parquet files directly from a URL using a simple `FROM` clause.</li>
        <li>Use the `read_parquet` function for more fine-grained control and efficiency.</li>
        <li>Load data from a Parquet file into a DuckDB table.</li>
        <li>Seamlessly analyze and visualize the data using SQL and Python.</li>
    </ul>

    DuckDB's native Parquet support makes it a powerful tool for interactive data analysis on large datasets without complex ETL pipelines.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    return mo, px


@app.cell
def _():
    import pyarrow
    return


if __name__ == "__main__":
    app.run()

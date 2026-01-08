# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "adbc-driver-sqlite==1.7.0",
#     "duckdb==1.4.0",
#     "lxml==6.0.0",
#     "marimo",
#     "pandas==2.3.2",
#     "polars==1.32.3",
#     "pyarrow==21.0.0",
#     "sqlalchemy==2.0.43",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loading Data

    _By [etrotta](https://github.com/etrotta)._

    This tutorial covers how to load data of varying formats and from different sources using [polars](https://docs.pola.rs/).

    It includes examples of how to load and write to a variety of formats, shows how to convert data from other libraries to support formats not supported directly by polars, includes relevant links for users that need to connect with external sources, and explains how to deal with custom formats via plugins.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pl):
    df = pl.DataFrame(
        [
            {"format": "Parquet", "lazy": True, "notes": None},
            {"format": "CSV", "lazy": True, "notes": None},
            {
                "format": "Databases",
                "lazy": False,
                "notes": "Requires another library as an Engine",
            },
            {
                "format": "Excel",
                "lazy": False,
                "notes": "Requires another library as an Engine",
            },
            {
                "format": "Newline-delimited JSON",
                "lazy": True,
                "notes": None,
            },
            {
                "format": "Traditional JSON",
                "lazy": False,
                "notes": None,
            },
            {"format": "Arrow", "lazy": False, "notes": "You can load XML and HTML files via pandas"},
            {"format": "Plugins", "lazy": True, "notes": "The most flexible, but takes some effort to implement"},
            {"format": "Feather / IPC", "lazy": True, "notes": None},
            {"format": "Avro", "lazy": False, "notes": None},
            {"format": "Delta", "lazy": True, "notes": "No Lazy writing"},
            {"format": "Iceberg", "lazy": True, "notes": "No Lazy writing"},
        ],
        orient="rows",
    )
    mo.vstack(
        [
            mo.ui.table(df, label="Quick Reference", pagination=False),
            "We will also use this table to demonstrate writing and reading to each format",
        ]
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parquet
    Parquet is a popular format for storing tabular data based on the Arrow memory spec, it is a great default and you'll find a lot of datasets already using it in sites like HuggingFace
    """)
    return


@app.cell
def _(df, folder, pl):
    df.write_parquet(folder / "data.parquet")  # Eager API - Writing to a file
    _ = pl.read_parquet(folder / "data.parquet")  # Eager API - Reading from a file
    lz = pl.scan_parquet(folder / "data.parquet")  # Lazy API - Reading from a file
    lz.sink_parquet(folder / "data_copy.parquet")  # Lazy API - Writing to a file
    return (lz,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CSV
    A classic and common format that has been widely used for decades.

    The API is almost identical to Parquet - You can just replace `parquet` by `csv` and it will work with the default settings, but polars also allows for you to customize some settings such as the delimiter and quoting rules.
    """)
    return


@app.cell
def _(df, folder, lz, pl):
    lz.sink_csv(folder / "data.csv")  # Lazy API - Writing to a file
    df.write_csv(folder / "data_no_head.csv", include_header=False, separator=";")  # Eager API - Writing to a file

    _ = pl.scan_csv(folder / "data.csv")  # Lazy API - Reading from a file
    _ = pl.read_csv(folder / "data_no_head.csv", has_header=False, separator=";")  # Eager API - Reading from a file
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## JSON

    JavaScript Object Notation is somewhat commonly used for storing unstructed data, and extremely commonly used for API responses.

    For large datasets you'll frequently see a variation in which each line in the file defines one separate object, called "Newline delimited JSON" (`ndjson`) or "JSON Lines" (`jsonl`)

    /// Note

        It's a lot more common to find Nested data in JSON than in other formats, but other formats such as Parquet also support nested datatypes.

        Polars supports Lists with variable length, Arrays with fixed length, and Structs with well defined fields, but not mappings with arbitrary keys.

        You might want to transform data by unnesting structs and exploding lists after loading from complex JSON files.
    """)
    return


@app.cell
def _(df, folder, lz, pl):
    # Newline Delimited JSON
    lz.sink_ndjson(folder / "data.ndjson")  # Lazy API - Writing to a file
    df.write_ndjson(folder / "data.ndjson")  # Eager API - Writing to a file

    _ = pl.scan_ndjson(folder / "data.ndjson")  # Lazy API - Reading from a file
    _ = pl.read_ndjson(folder / "data.ndjson")  # Eager API - Reading from a file

    # Normal JSON
    df.write_json(folder / "data.json")  # Eager API - Writing to a file
    _ = pl.read_json(folder / "data.json")  # Eager API - Reading from a file

    # Note that there are no Lazy methods for normal JSON files,
    # either use NDJSON instead or use `lz.collect().write_json()` to collect into memory before writing, and `pl.read_json().lazy()` to read into memory before operating in lazy mode
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Databases

    Polars doesn't supports any databases _directly_, but rather uses other libraries as Engines. Reading and writing to databases using polars methods does not supports Lazy execution, but you may pass an SQL Query for the database to pre-filter the data before reaches polars. See the [User Guide](https://docs.pola.rs/user-guide/io/database)  for more details.

    You can also use other libraries with [arrow support](#arrow-support) or [polars plugins](#plugin-support) to read from databases before loading into polars, some of which support lazy reading.

    Using the Arrow Database Connectivity SQLite support as an example:
    """)
    return


@app.cell
def _(df, folder, pl):
    URI = "sqlite:///" + f"/{folder.resolve()}/db.sqlite"
    df.write_database(table_name="quick_reference", connection=URI, engine="adbc", if_table_exists="replace")

    query = """SELECT * FROM quick_reference WHERE format LIKE '%Database%'"""

    pl.read_database_uri(query=query, uri=URI, engine="adbc")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Excel

    From a performance perspective, we recommend using other formats if possible, such as Parquet or CSV files.

    Similarly to Databases, polars doesn't supports it natively but rather uses other libraries as Engines. See the [User Guide](https://docs.pola.rs/user-guide/io/excel) if you need to use it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Others natively supported

    If you understood the above examples, then all other formats should feel familiar - the core API is the same for all formats, `read` and `write` for the Eager API or `scan` and `sink` for the lazy API.

    See https://docs.pola.rs/api/python/stable/reference/io.html for the full list of formats natively supported by Polars
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arrow Support

    You can convert Arrow compatible data from other libraries such as `pandas`, `duckdb` or `pyarrow` to polars DataFrames and vice-versa, much of the time without even having to copy data.

    This allows for you to use other libraries to load data in formats not support by polars, then convert the dataframe in-memory to polars.
    """)
    return


@app.cell
def _(df, folder, pd, pl):
    # XML Example using `pandas` read_xml() and to_xml() methods
    df.to_pandas().to_xml(folder / "data.xml")
    pandas_df = pd.read_xml(folder / "data.xml")
    _ = pl.from_pandas(pandas_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plugin Support

    You can also write [IO Plugins](https://docs.pola.rs/user-guide/plugins/io_plugins/) for Polars in order to support any format you need, or use other libraries that support polars via their own plugins such as DuckDB.
    """)
    return


@app.cell
def _(duckdb, folder):
    # Requires duckdb >= 1.4.0
    conn = duckdb.connect(folder / "db.sqlite")
    conn.sql("SELECT * FROM quick_reference").pl(lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating your own Plugin

    The simplest form of plugins are essentially generators that yield DataFrames.

    Without parsing filters you will be missing on performance improvements, but even just this can help improve your performance in many cases as it allows for polars to optimize the query and request data in batches as opposed to always loading everything in memory.

    Below is a example plugin which just takes the product between multiple iterables, some highlights are that:

    - You must use `register_io_source` for polars to create the LazyFrame which will consume the Generator
    - You are expected to provide a Schema before the Generator starts
    - - For many use cases the Plugin may be able to infer it, but you could also pass it explicitly to the plugin function
    - Ideally you should parse some of the filters and column selectors to avoid unnecessary work, but it is possible to delegate that to polars after loading the data in order to keep it simpler (at the cost of efficiency)

    Efficiently parsing the filter expressions is out of the scope for this notebook.
    """)
    return


@app.cell
def _(my_custom_input_plugin):
    my_custom_input_plugin(int, range(3), range(5))
    return


@app.cell
def _(my_custom_input_plugin, pl):
    my_custom_input_plugin(bool, [True, False], [True, False]).with_columns(
        (pl.col("A") & pl.col("B")).alias("AND"),
        (pl.col("A") & pl.col("B")).not_().alias("NAND"),
        (pl.col("A") | pl.col("B")).alias("OR"),
        (pl.col("A") ^ pl.col("B")).alias("XOR"),
    ).collect()
    return


@app.cell
def _(Iterator, get_positional_names, itertools, pl, register_io_source):
    def my_custom_input_plugin(dtype, *iterables) -> pl.LazyFrame:
        schema = pl.Schema({key: dtype for key in get_positional_names(len(iterables))})

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            """
            Generator function that creates the source.
            This function will be registered as IO source.
            """
            if batch_size is None:
                batch_size = 100
            if n_rows is not None:
                batch_size = min(batch_size, n_rows)

            generator = itertools.product(*iterables)
            while n_rows is None or n_rows > 0:
                rows = []
                try:
                    while len(rows) < batch_size:
                        rows.append(next(generator))
                except StopIteration:
                    n_rows = -1

                df = pl.from_records(rows, schema=schema, orient="row")
                if n_rows is not None:
                    n_rows -= df.height
                    batch_size = min(batch_size, n_rows)

                # If we would make a performant reader, we would not read these
                # columns at all.
                if with_columns is not None:
                    df = df.select(with_columns)

                # If the source supports predicate pushdown, the expression can be parsed
                # to skip rows/groups.
                if predicate is not None:
                    df = df.filter(predicate)

                yield df

        return register_io_source(io_source=source_generator, schema=schema)
    return (my_custom_input_plugin,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DuckDB

    As demonstrated above, in addition to Arrow interoperability support, [DuckDB](https://duckdb.org/) also has added support for loading query results into a polars DataFrame or LazyFrame via a polars plugin.

    You can read more about polars and duckdb integrations in

    - https://docs.pola.rs/user-guide/ecosystem/#duckdb
    - https://duckdb.org/docs/stable/guides/python/polars.html

    You can learn more about DuckDB in the marimo course about it as well, including Marimo SQL related features
    """)
    return


@app.cell
def _():
    # Amazing if you need of features not yet supported by Polars such as geospatial data
    duckdb_query = """
        SELECT 
            id,
            name,
            ST_X(geometry) as longitude,
            ST_Y(geometry) as latitude
        FROM locations
    """
    return (duckdb_query,)


@app.cell
def _(duckdb_conn, duckdb_query):
    # Eager (default):
    duckdb_conn.sql(duckdb_query).pl()
    return


@app.cell
def _(duckdb_conn, duckdb_query):
    # Lazy (requires >= 1.4.0):
    duckdb_conn.sql(duckdb_query).pl(lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hive Partitions

    There is also support for [Hive](https://docs.pola.rs/user-guide/io/hive/) partitioned data, but parts of the API are still unstable (may change in future polars versions
    ).

    Even without using partitions, many methods also support glob patterns to read multiple files in the same folder such as `scan_csv(folder / "*.csv")`
    """)
    return


@app.cell
def _(df, folder, pl):
    df.write_parquet(str((folder / "hive").resolve()) + "/", partition_by=["lazy"])
    _ = pl.scan_parquet(str((folder / "hive").resolve()) + "/").filter(pl.col("lazy").eq(True)).collect()

    print(*(folder / "hive").rglob("*.parquet"), sep="\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reading from the Cloud

    Polars also has support for reading public and private datasets from multiple websites
    and cloud storage solutions.

    If you must (re)use the same file many times in the same machine you may want to manually download it then load from your local file system instead to avoid re-downloading though, or download and write to disk only if the file does not exists.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arbitrary web sites

    You can load files from nearly any website just by using a HTTPS URL, as long as it is not locked behind authorization.
    """)
    return


@app.cell(disabled=True)
def _():
    # df = pl.read_csv('https://example.com/file.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hugging Face & Kaggle Datasets

    Look for polars inside of dropdowns such as "Use this dataset" in Hugging Face or "Code" in Kaggle, and oftentimes you'll get a snippet to load data directly into a dataframe you can use

    Read more: [Hugging Face](https://docs.pola.rs/user-guide/io/hugging-face/), [Kaggle](https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpolars)
    """)
    return


@app.cell(disabled=True)
def _():
    # df = pl.read_parquet('hf://datasets/username/dataset/*.parquet')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cloud Storage - AWS S3, Azure Blob Storage, Google Cloud Storage

    The API is the same for all three storage providers, check the [User Guide](https://docs.pola.rs/user-guide/io/cloud-storage/) if you need of any of them.

    Runnable examples are not included in this Notebook as it would require setting up authentication, but the disabled cell below shows an example using Azure.
    """)
    return


@app.cell(disabled=True)
def _(adlfs, df, os, pl):
    fs = adlfs.AzureBlobFileSystem(connection_string=os.environ["AZURE_STORAGE_CONNECTION_STRING"])
    destination = f"abfs://{os.environ['AZURE_CONTAINER_NAME']}/file.parquet"

    # Writing
    with fs.open(destination, mode="wb") as f:
        df.write_parquet(f)

    # Reading
    pl.read_parquet(
        destination, storage_options={"account_name": os.environ["AZURE_STORAGE_ACCOUNT"], "use_azure_cli": "True"}
    )

    # Deleting
    fs.delete(destination)

    # If you get an error saying that the account does not exists, double check you logged in the correct account and subscription via `az login`
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multiplexing

    You can also split a query into multiple sinks via [multiplexing](https://docs.pola.rs/user-guide/lazy/multiplexing/), to avoid reading multiple times, repeating the same operations for each sink or collecting intermediary results into memory.
    """)
    return


@app.cell
def _(folder, lz, pl):
    lz2 = lz.with_columns(pl.col(pl.String).str.to_uppercase())
    lz3 = lz.with_columns(pl.col(pl.String).str.to_lowercase())

    # Collecting multiple LazyFrames into memory
    _df, _df2, _df3 = pl.collect_all([lz, lz2, lz3])

    # Sinking multiple LazyFrames into different destinations
    sinks = [
        lz.sink_csv(folder / "data_1.csv", lazy=True),
        lz2.sink_csv(folder / "data_2.csv", lazy=True),
        lz3.sink_csv(folder / "data_3.csv", lazy=True),
    ]
    _ = pl.collect_all(sinks)
    return (sinks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Async Execution

    Polars also has experimental support for running lazy queries in `async` mode, letting you `await` operations inside of async functions.
    """)
    return


@app.cell
async def _(lz):
    await lz.collect_async()
    return


@app.cell
async def _(folder, lz, pl, sinks):
    # If you want to write to a file, use `lz.sink_format(lazy=True)` followed by `...collect_async()` or `pl.collect_all_async(...)`
    _ = await lz.sink_csv(folder / "data_from_async.csv", lazy=True).collect_async()
    _ = await pl.collect_all_async(sinks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion
    As you have seen, polars makes it easy to work with a variety of formats and different data sources.

    From natively supported formats such as Parquet and CSV files, to using other libraries as an intermediary for XML or geospatial data, and plugins for newly emerging or proprietary formats, as long as your data can fit in a table then odds are you can turn it into a polars DataFrame.

    Combined with loading directly from remote sources, including public data platforms such as Hugging Face and Kaggle as well as private data in your cloud, you can import datasets for almost anything you can imagine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utilities
    Imports, utility functions and alike used through the Notebook
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(disabled=True)
def _():
    # You may need to install `fsspec ` and `adlfs ` beyond the dependencies included in the notebook
    import os
    import adlfs
    return adlfs, os


@app.cell
def _():
    import pathlib
    import tempfile

    folder = pathlib.Path(tempfile.mkdtemp())
    folder
    return (folder,)


@app.cell
def _():
    import math
    import string
    import itertools
    from typing import Iterator
    return Iterator, itertools, string


@app.cell
def _():
    import polars as pl
    import pandas as pd
    from polars.io.plugins import register_io_source
    import duckdb
    return duckdb, pd, pl, register_io_source


@app.cell
def _(itertools, string):
    def get_positional_names(count: int) -> list[str]:
        out = []
        size = 0
        while True:
            size += 1  # number of characters in each column name
            for column in itertools.product(*itertools.repeat(string.ascii_uppercase, size)):
                if len(out) >= count:
                    return out
                out.append("".join(column))
    return (get_positional_names,)


@app.cell
def _(duckdb):
    # Connect to an ephemeral in-memory DuckDB database
    duckdb_conn = duckdb.connect(":memory:")

    # Install and load the spatial extension for geometry support
    duckdb_conn.install_extension("spatial")
    duckdb_conn.load_extension("spatial")

    # Create a table with geometry column
    duckdb_conn.sql("""
        CREATE TABLE locations (
            id INTEGER,
            name VARCHAR,
            geometry GEOMETRY
        )
    """)

    # Insert some sample data with geometry points
    duckdb_conn.sql("""
        INSERT INTO locations VALUES
        (1, 'New York', ST_Point(-74.0059, 40.7128)),
        (2, 'Los Angeles', ST_Point(-118.2437, 34.0522)),
        (3, 'Chicago', ST_Point(-87.6298, 41.8781)),
        (4, 'Houston', ST_Point(-95.3698, 29.7604)),
        (5, 'Phoenix', ST_Point(-112.0740, 33.4484))
    """)
    return (duckdb_conn,)


if __name__ == "__main__":
    app.run()

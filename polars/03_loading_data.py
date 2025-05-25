# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "adbc-driver-sqlite==1.6.0",
#     "lxml==5.4.0",
#     "marimo",
#     "pandas==2.2.3",
#     "polars==1.30.0",
#     "pyarrow==20.0.0",
#     "sqlalchemy==2.0.41",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Loading Data

    _By [etrotta](https://github.com/etrotta)._

    This tutorial covers how to load data of varying formats and from different sources using [polars](https://docs.pola.rs/).

    It includes examples of how to load and write to a variety of formats, shows how to convert data from other libraries to support formats not supported directly by polars, includes relevants links for users that need to connect with external sources, and explains how to deal with custom formats via plugins.
    """
    )
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
            {"format": "Plugins", "lazy": True, "notes": "The most flexibile, but takes some effort to implement"},
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
    mo.md(
        r"""
    ## Parquet
    Parquet is a popular format for storing tabular data based on the Arrow memory spec, it is a great default and you'll find a lot of datasets already using it in sites like HuggingFace
    """
    )
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
    mo.md(
        r"""
    ## CSV
    A classic and common format that has been widely used for decades.

    The API is almost identic to Parquet - You can just replace `parquet` by `csv` and it will work with the default settings, but polars also allows for you to customize some settings such as the delimiter and quoting rules.
    """
    )
    return


@app.cell
def _(df, folder, lz, pl):
    lz.sink_csv(folder / "data.csv")  # Lazy API - Writing to a file
    df.write_csv(folder / "data_no_head.csv", include_header=False, separator=",")  # Eager API - Writing to a file

    _ = pl.scan_csv(folder / "data.csv")  # Lazy API - Reading from a file
    _ = pl.read_csv(folder / "data_no_head.csv", has_header=False, separator=";")  # Eager API - Reading from a file
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## JSON

    JavaScript Object Notation is somewhat commonly used for storing unstructed data, and extremely commonly used for API responses.

    For large datasets you'll frequently see a variation in which each line in the file defines one separate object, called "Newline delimited JSON" (`ndjson`) or "JSON Lines" (`jsonl`)

    /// Note

        It's a lot more common to find Nested data in JSON than in other formats, but other formats such as Parquet also support nested datatypes.

        Polars supports Lists with variable length, Arrays with fixed length, and Structs with well defined fields, but not mappings with arbitrary keys.

        You might want to transform data using by unnested structs and exploding lists after loading from complex JSON files.
    """
    )
    return


@app.cell
def _(df, folder, lz, pl):
    # Newline Delimited JSON
    lz.sink_ndjson(folder / "data.ndjson")  # Lazy API - Writing to a file
    df.write_ndjson(folder / "data.ndjson")  # Eager API - Writing to a file

    _ = pl.scan_ndjson(folder / "data.ndjson")  # Lazy API - Reading from a file
    _ = pl.read_ndjson(folder / "data.ndjson")  # Eager API - Reading from a file

    # Normal JSON
    df.write_json(folder / "data_no_head.json")  # Eager API - Writing to a file
    _ = pl.read_json(folder / "data_no_head.json")  # Eager API - Reading from a file

    # Note that there are no Lazy methods for normal JSON files,
    # either use NDJSON instead or use `lz.collect().write_json()` to collect into memory before writing, and `pl.read_json().lazy()` to read into memory before operating in lazy mode
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Databases

    Polars doesn't supports any _directly_, but rather uses other libraries as Engines. Reading and writing to databases does not supports Lazy execution, but you may pass an SQL Query for the database to pre-filter the data before reaches polars. See the [User Guide](https://docs.pola.rs/user-guide/io/database)  for more details.

    Using the Arrow Database Connectivity SQLite support as an example:
    """
    )
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
    mo.md(
        r"""
    ## Excel

    From a performance perspective, we recommend using other formats if possible, such as Parquet or CSV files.

    Similarly to Databases, polars doesn't supports it natively but rather uses other libraries as Engines. See the [User Guide](https://docs.pola.rs/user-guide/io/excel) if you need to use it.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Others natively supported

    If you understood the above examples, then all other formats should feel familiar - the core API is the same for all formats, `read` and `write` for the Eager API or `scan` and `sink` for the lazy API.

    See https://docs.pola.rs/api/python/stable/reference/io.html for the full list of formats natively supported by Polars
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Arrow Support

    You can convert Arrow compatible data from other libraries such as `pandas`, `duckdb` or `pyarrow` to polars DataFrames and vice-versa, much of the time without even having to copy data.

    This allows for you to use other libraries to load data in formats not support by polars, then convert the dataframe in-memory to polars.
    """
    )
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
    mo.md(
        r"""
    ## Plugin Support

    You can also write [IO Plugins](https://docs.pola.rs/user-guide/plugins/io_plugins/) for Polars in order to support any format you need.

    TODO UPDATE THIS SECTION

    - Consider whenever we want to include a full example
    - Link an example of a real production-grade plugin
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ## Hive Partitions

    There is also support for [Hive](https://docs.pola.rs/user-guide/io/hive/) partitioned data, but parts of the API are still unstable (may change in future polars versions
    ).

    Even without using partitions, many methods also support glob patterns to read multiple files in the same folder such as `scan_csv(folder / "*.csv")`
    """
    )
    return


@app.cell
def _(df, folder, pl):
    df.write_parquet(str((folder / "hive").resolve()) + "/", partition_by=["lazy"])
    _ = pl.scan_parquet(str((folder / "hive").resolve()) + "/").filter(pl.col("lazy").eq(True)).collect()

    print(*(folder / "hive").rglob("*.parquet"), sep="\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Reading from the Cloud

    Polars also has support for reading public and private datasets from multiple websites 
    and cloud storage solutions.

    If you must (re)use the same file many times in the same machine you may want to manually download it then load from your local file system instead to avoid re-downloading though, or download and write to disk only if the file does not exists.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Arbitrary web sites

    You can load files from nearly any website just by using a HTTPS URL, as long as it is not locked behind authorization.
    """
    )
    return


@app.cell(disabled=True)
def _():
    # df = pl.read_csv('https://example.com/file.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Hugging Face & Kaggle Datasets

    Look for polars inside of dropdowns such as "Use this dataset" in Hugging Face or "Code" in Kaggle, and oftentimes you'll get a snippet to load data directly into a dataframe you can use

    Read more: [Hugging Face](https://docs.pola.rs/user-guide/io/hugging-face/), [Kaggle](https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpolars)
    """
    )
    return


@app.cell(disabled=True)
def _():
    # df = pl.read_parquet('hf://datasets/username/dataset/*.parquet')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Cloud Storage - AWS S3, Azure Blob Storage, Google Cloud Storage

    The API is the same for all three storage providers, check the [User Guide](https://docs.pola.rs/user-guide/io/cloud-storage/) if you need of any of them.

    Examples are not included in this Notebook as it would require setting up authentication.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pathlib
    import tempfile

    folder = pathlib.Path(tempfile.mkdtemp())
    folder
    return (folder,)


@app.cell
def _():
    import polars as pl
    import pandas as pd
    return pd, pl


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb==1.2.1",
#     "marimo",
#     "polars[pyarrow]==1.25.2",
#     "sqlglot==26.11.1",
# ]
# ///
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "duckdb==1.2.1",
#     "sqlglot==26.11.1",
#     "polars[pyarrow]==1.25.2",
# ]
# ///

import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Loading JSON

        DuckDB supports reading and writing JSON through the `json` extension that should be present in most distributions and is autoloaded on first-use. If it's not, you can [install and load](https://duckdb.org/docs/stable/data/json/installing_and_loading.html) it manually like any other extension.

        In this tutorial we'll cover 4 different ways we can transfer JSON data in and out of DuckDB:

        - [`FROM`](https://duckdb.org/docs/stable/sql/query_syntax/from.html) statement.
        - [`read_json`](https://duckdb.org/docs/stable/data/json/loading_json#the-read_json-function) function.
        - [`COPY`](https://duckdb.org/docs/stable/sql/statements/copy#copy--from) statement.
        - [`IMPORT DATABASE`](https://duckdb.org/docs/stable/sql/statements/export.html) statement.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using `FROM`

        Loading data using `FROM` is simple and straightforward. We use a path or URL to the file we want to load where we'd normally put a table name. When we do this, DuckDB attempts to infer the right way to read the file including the correct format and column types. In most cases this is all we need to load data into DuckDB.
        """
    )
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        SELECT * FROM 'https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/cars.json';
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using `read_json`

        For greater control over how the JSON is read, we can directly call the [`read_json`](https://duckdb.org/docs/stable/data/json/loading_json#the-read_json-function) function. It supports a few different arguments — some common ones are:

        - `format='array'` or `format='newline_delimited'` - the former tells DuckDB that the rows should be read from a top-level JSON array while the latter means the rows should be read from JSON objects separated by a newline (JSONL/NDJSON).
        - `ignore_errors=true` - skips lines with parse errors when reading newline delimited JSON.
        - `columns={columnName: type, ...}` - lets you set types for individual columns manually.
        - `dateformat` and `timestampformat` - controls how DuckDB attempts to parse [Date](https://duckdb.org/docs/stable/sql/data_types/date) and [Timestamp](https://duckdb.org/docs/stable/sql/data_types/timestamp) types. Use the format specifiers specified in the [docs](https://duckdb.org/docs/stable/sql/functions/dateformat.html#format-specifiers).

        We could rewrite the previous query more explicitly as:
        """
    )
    return


@app.cell
def _(mo):
    cars_df = mo.sql(
        f"""
        SELECT *
        FROM
            read_json(
                'https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/cars.json',
                format = 'array',
                columns = {{
                    Name:'VARCHAR',
                    Miles_per_Gallon:'FLOAT',
                    Cylinders:'FLOAT',
                    Displacement:'FLOAT',
                    Horsepower:'FLOAT',
                    Weight_in_lbs:'FLOAT',
                    Acceleration:'FLOAT',
                    Year:'DATE',
                    Origin:'VARCHAR'
                }},
                dateformat = '%Y-%m-%d'
            )
        ;
        """
    )
    return (cars_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Other than singular files we can read [multiple files](https://duckdb.org/docs/stable/data/multiple_files/overview.html) at a time by either passing a list of files or a UNIX glob pattern.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using `COPY`

        `COPY` is for useful both for importing and exporting data in a variety of formats including JSON. For example, we can import data into an existing table from a JSON file.
        """
    )
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        CREATE OR REPLACE TABLE cars2 (
            Name VARCHAR,
            Miles_per_Gallon VARCHAR,
            Cylinders VARCHAR,
            Displacement FLOAT,
            Horsepower FLOAT,
            Weight_in_lbs FLOAT,
            Acceleration FLOAT,
            Year DATE,
            Origin VARCHAR
        );
        """
    )
    return (cars2,)


@app.cell
def _(cars2, mo):
    _df = mo.sql(
        f"""
        COPY cars2 FROM 'https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/cars.json' (FORMAT json, ARRAY true, DATEFORMAT '%Y-%m-%d');
        SELECT * FROM cars2;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Similarly, we can write data from a table or select statement to a JSON file. For example, we create a new JSONL file with just the car names and miles per gallon. We first create a temporary directory to avoid cluttering our project directory.""")
    return


@app.cell
def _(Path):
    from tempfile import TemporaryDirectory

    TMP_DIR = TemporaryDirectory()
    COPY_PATH = Path(TMP_DIR.name) / "cars_mpg.jsonl"
    print(COPY_PATH)
    return COPY_PATH, TMP_DIR, TemporaryDirectory


@app.cell
def _(COPY_PATH, cars2, mo):
    _df = mo.sql(
        f"""
        COPY (
            SELECT 
                Name AS car_name,
                "Miles_per_Gallon" AS mpg 
            FROM cars2 
            WHERE mpg IS NOT null
        ) TO '{COPY_PATH}' (FORMAT json);
        """
    )
    return


@app.cell
def _(COPY_PATH, Path):
    Path(COPY_PATH).exists()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using `IMPORT DATABASE`

        The last method we can use to load JSON data is using the `IMPORT DATABASE` statement. It works in conjunction with `EXPORT DATABASE` to save and load an entire database to and from a directory. For example let's try and export our default in-memory database.
        """
    )
    return


@app.cell
def _(Path, TMP_DIR):
    EXPORT_PATH = Path(TMP_DIR.name) / "cars_export"
    print(EXPORT_PATH)
    return (EXPORT_PATH,)


@app.cell
def _(EXPORT_PATH, mo):
    _df = mo.sql(
        f"""
        EXPORT DATABASE '{EXPORT_PATH}' (FORMAT json);
        """
    )
    return


@app.cell
def _(EXPORT_PATH, Path):
    list(Path(EXPORT_PATH).iterdir())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can then load the database back into DuckDB.""")
    return


@app.cell
def _(EXPORT_PATH, mo):
    _df = mo.sql(
        f"""
        DROP TABLE IF EXISTS cars2;
        IMPORT DATABASE '{EXPORT_PATH}';
        SELECT * FROM cars2;
        """
    )
    return


@app.cell(hide_code=True)
def _(TMP_DIR):
    TMP_DIR.cleanup()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further Reading

        - Complete information on the JSON support in DuckDB can be found in their [documentation](https://duckdb.org/docs/stable/data/json/overview.html).
        - You can also learn more about using SQL in marimo from the [examples](https://github.com/marimo-team/marimo/tree/main/examples/sql).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    return Path, mo


if __name__ == "__main__":
    app.run()

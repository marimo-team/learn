import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Basic Queries

    As a running example, we will replicate parts of the analysis in the
    paper, "Off the beaten path: Gaia reveals GD-1 stars outside
    of the main stream" by Adrian Price-Whelan and Ana Bonaca.

    ## Query Language

    In order to select data from a database, you need to compose a query,
    which is a program written in a "query language".
    The query language we will use is ADQL, which stands for "Astronomical
    Data Query Language".

    ADQL is a dialect of SQL (Structured Query Language), which is by far
    the most commonly used query language. Almost everything you will learn
    about ADQL also works in SQL.

    ## Connecting to Gaia

    The library we will use to get Gaia data is Astroquery.
    Astroquery provides `Gaia`, which is an object that represents a
    connection to the Gaia database.
    """)
    return


@app.cell
def _():
    from astroquery.gaia import Gaia
    return (Gaia,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Databases and Tables

    We can use `Gaia.load_tables` to get the names of the tables in the
    Gaia database. With the option `only_names=True`, it loads
    information about the tables (metadata) but not the data itself.
    """)
    return


@app.cell
def _(Gaia):
    tables = Gaia.load_tables(only_names=True)
    for table in tables:
        print(table.name)
    return (tables,)


@app.cell
def _(mo):
    mo.md(r"""
    We can use `load_table` (not `load_tables`) to get the metadata for a
    single table. Note that this only downloads metadata, not the table contents.
    """)
    return


@app.cell
def _(Gaia):
    table_metadata = Gaia.load_table('gaiadr2.gaia_source')
    print(table_metadata)
    return (table_metadata,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Columns

    The following loop prints the names of the columns in the table.
    """)
    return


@app.cell
def _(table_metadata):
    for column in table_metadata.columns:
        print(column.name)
    return (column,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (2 minutes)

    One of the other tables we will use is `gaiadr2.panstarrs1_original_valid`.
    Use `load_table` to get the metadata for this table. How many columns are
    there and what are their names?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    ```python
    panstarrs_metadata = Gaia.load_table('gaiadr2.panstarrs1_original_valid')
    print(panstarrs_metadata)

    for column in panstarrs_metadata.columns:
        print(column.name)
    ```

    There are 26 columns, including: obj_name, obj_id, ra, dec, ra_error,
    dec_error, epoch_mean, g_mean_psf_mag, g_mean_psf_mag_error, g_flags,
    r_mean_psf_mag, r_mean_psf_mag_error, r_flags, i_mean_psf_mag,
    i_mean_psf_mag_error, i_flags, z_mean_psf_mag, z_mean_psf_mag_error,
    z_flags, y_mean_psf_mag, y_mean_psf_mag_error, y_flags, n_detections,
    zone_id, obj_info_flag, quality_flag.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Writing Queries

    A query is a program written in a query language like SQL. For the Gaia
    database, the query language is a dialect of SQL called ADQL.

    Here's an example of an ADQL query using a triple-quoted string so we can
    include line breaks, which makes it easier to read.
    """)
    return


@app.cell
def _():
    query1 = """SELECT
TOP 10
source_id, ra, dec, parallax
FROM gaiadr2.gaia_source
"""
    return (query1,)


@app.cell
def _(Gaia, query1):
    job1 = Gaia.launch_job(query1)
    print(job1)
    return (job1,)


@app.cell
def _(job1):
    results1 = job1.get_results()
    results1
    return (results1,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (3 minutes)

    Read the documentation of this table and choose a column that looks
    interesting to you. Add the column name to the query and run it again.
    What are the units of the column you selected? What is its data type?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    For example, we can add `radial_velocity`:

    ```python
    query1_with_rv = """SELECT
    TOP 10
    source_id, ra, dec, parallax, radial_velocity
    FROM gaiadr2.gaia_source
    """
    job1_with_rv = Gaia.launch_job(query1_with_rv)
    results1_with_rv = job1_with_rv.get_results()
    results1_with_rv
    ```

    `radial_velocity` has units of km/s and data type float64.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Asynchronous Queries

    `launch_job` asks the server to run the job "synchronously", but synchronous
    jobs are limited to 2000 rows. For queries that return more rows, you should
    run "asynchronously" using `launch_job_async`.

    The following query selects 3000 rows with two additional columns (`pmra`
    and `pmdec`) and a `WHERE` clause to filter by parallax.
    """)
    return


@app.cell
def _():
    query2 = """SELECT
TOP 3000
source_id, ra, dec, pmra, pmdec, parallax
FROM gaiadr2.gaia_source
WHERE parallax < 1
"""
    return (query2,)


@app.cell
def _(Gaia, query2):
    job2 = Gaia.launch_job_async(query2)
    job2
    return (job2,)


@app.cell
def _(job2):
    results2 = job2.get_results()
    results2
    return (results2,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    The clauses in a query have to be in the right order. Go back and
    change the order of the clauses in `query2` and run it again.
    The modified query should fail, but notice that you don't get much
    useful debugging information.

    For this reason, developing and debugging ADQL queries can be really hard.
    A few suggestions:

    - Start with a working query and make small changes.
    - Use `TOP` to limit the number of rows while testing.
    - Launch test queries synchronously to make them start faster.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    If you put the WHERE clause before FROM, the query will fail:

    ```python
    query2_erroneous = """SELECT
    TOP 3000
    source_id, ra, dec, pmra, pmdec, parallax
    WHERE parallax < 1
    FROM gaiadr2.gaia_source
    """
    ```

    The required clause order is: SELECT ... FROM ... WHERE ...
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Operators

    In a `WHERE` clause, you can use SQL comparison operators:

    | Symbol | Operation             |
    | ------ | :-------------------- |
    | `>`    | greater than          |
    | `<`    | less than             |
    | `>=`   | greater than or equal |
    | `<=`   | less than or equal    |
    | `=`    | equal                 |
    | `!=` or `<>` | not equal       |

    You can combine comparisons using `AND`, `OR`, and `NOT`.
    Note that the equality operator is `=`, not `==` as in Python.

    ## Formatting Queries

    It is often better to write Python code that assembles a query for you.
    One useful tool is the string `format` method. Here we divide the query
    into a list of column names and a base query with format specifiers.
    """)
    return


@app.cell
def _():
    columns = 'source_id, ra, dec, pmra, pmdec, parallax'

    query3_base = """SELECT
TOP 10
{columns}
FROM gaiadr2.gaia_source
WHERE parallax < 1
  AND bp_rp BETWEEN -0.75 AND 2
"""
    return (columns, query3_base)


@app.cell
def _(columns, query3_base):
    query3 = query3_base.format(columns=columns)
    print(query3)
    return (query3,)


@app.cell
def _(Gaia, query3):
    job3 = Gaia.launch_job(query3)
    print(job3)
    return (job3,)


@app.cell
def _(job3):
    results3 = job3.get_results()
    results3
    return (results3,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (10 minutes)

    This query always selects sources with `parallax` less than 1. But
    suppose you want to take that upper bound as an input.

    Modify `query3_base` to replace `1` with a format specifier like
    `{max_parallax}`. Now, when you call `format`, add a keyword argument
    that assigns a value to `max_parallax`, and confirm that the format
    specifier gets replaced with the value you provide.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    ```python
    query_base_sol = """SELECT
    TOP 10
    {columns}
    FROM gaiadr2.gaia_source
    WHERE parallax < {max_parallax} AND
    bp_rp BETWEEN -0.75 AND 2
    """

    query_sol = query_base_sol.format(columns=columns,
                              max_parallax=0.5)
    print(query_sol)
    ```
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    This episode demonstrated:

    1. Making a connection to the Gaia server
    2. Exploring information about the database and the tables it contains
    3. Writing a query and sending it to the server
    4. Downloading the response from the server as an Astropy `Table`

    Key points:
    - Use queries to select only the data you need from large databases
    - Read the metadata and documentation to understand the tables and columns
    - Develop queries incrementally: start simple, test, and add a little at a time
    - Use `TOP` and `COUNT` to test before running queries that return lots of data
    - If a query returns fewer than 3000 rows, run it synchronously; otherwise, asynchronously
    """)
    return


if __name__ == "__main__":
    app.run()

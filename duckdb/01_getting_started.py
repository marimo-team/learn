import marimo

__generated_with = "0.11.30"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _introduction(mo):
    mo.md(
        """
        # DuckDB: An Embeddable Analytical Database System

        ### What is DuckDB?

        [DuckDB](https://duckdb.org/) is a high-performance, in-process analytical database management system (DBMS) designed for speed and simplicity. It's particularly well-suited for analytical query workloads, offering a robust SQL interface and efficient data processing capabilities. This document highlights key features and aspects of DuckDB relevant for a course on database systems or data analysis.

        ### [Key Features](https://duckdb.org/why_duckdb):

        - In-Process: Easy integration, zero external dependencies.
        - Portable: Works on various OS and architectures.
        - Columnar Storage: Efficient for analytical queries.
        - Vectorized Execution: Speeds up data processing.
        - ACID Transactions: Ensures data integrity.
        - Multi-Language APIs: Python, R, Java, etc.

        ### [Use Cases](https://github.com/davidgasquez/awesome-duckdb?tab=readme-ov-file):

        - Data analysis and exploration
        - Embedded analytics in applications
        - ETL (Extract, Transform, Load) processes
        - Data science and machine learning workflows
        - Rapid prototyping of data analysis pipelines.

        ### [Installation](https://duckdb.org/docs/installation/?version=stable&environment=python):

        - The DuckDB Python API can be installed using pip:
        ```
        pip install duckdb
        ```
        - It is also possible to install DuckDB using conda:
        ```
        conda install python-duckdb -c conda-forge.
        ```

        **Python version:** DuckDB requires Python 3.7 or newer.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # [1. DuckDB Basic Connection](https://duckdb.org/docs/stable/connect/overview.html)

        DuckDB can run entirely in your computer's RAM, known as in-memory mode, which you can enable by using `:memory:` as the database name or by not providing a database file. It's crucial to understand that this means all data is temporary and will be completely erased when the program closes, as it isn't saved to disk.
        """
    )
    return


@app.cell
def _database_connection(duckdb):
    # Create a connection to an in-memory database
    database_connection = duckdb.connect(database=":memory:")
    print(
        f"DuckDB version: {database_connection.execute('SELECT version()').fetchone()[0]}"
    )
    return (database_connection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""# [2. Creating Tables](https://duckdb.org/docs/stable/sql/statements/create_table.html)"""
    )
    return


@app.cell
def _create_users_table(database_connection):
    database_connection.execute(
        """
    CREATE TABLE users (
        id INTEGER,
        name VARCHAR,
        age INTEGER,
        registration_date DATE
    )
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""# [3. Instering data into table](https://duckdb.org/docs/stable/sql/statements/insert)"""
    )
    return


@app.cell
def _insert_user_data(database_connection):
    database_connection.execute(
        """
    INSERT INTO users VALUES
    (1, 'Alice', 25, '2021-01-01'),
    (2, 'Bob', 30, '2021-02-01'),
    (3, 'Charlie', 35, '2021-03-01')
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""# [4. Basic Queries](https://duckdb.org/docs/stable/sql/query_syntax/select)"""
    )
    return


@app.cell
def _basic_queries(database_connection):
    # Select all data
    user_results = database_connection.execute("SELECT * FROM users").fetchall()
    for user_row in user_results:
        print(user_row)
    return user_results, user_row


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""# [5. Working with Polars](https://duckdb.org/docs/stable/guides/python/polars.html)"""
    )
    return


@app.cell
def _polars_dataframe(database_connection, pl):
    # Create a Polars DataFrame
    polars_dataframe = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "registration_date": ["2021-01-01", "2021-02-01", "2021-03-01"],
        }
    )

    # Register the Polars DataFrame as a DuckDB table
    database_connection.register("users_polars", polars_dataframe)

    # Query the Polars DataFrame using DuckDB
    polars_results = database_connection.execute(
        "SELECT * FROM users_polars"
    ).fetchall()
    print("New Table:")
    for polars_row in polars_results:
        print(polars_row)
    return polars_dataframe, polars_results, polars_row


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""# [6. Join Operations](https://duckdb.org/docs/stable/guides/performance/join_operations.html)"""
    )
    return


@app.cell
def _join_operations(database_connection):
    join_results = database_connection.execute(
        """
    SELECT u.id, u.name, u.age, nu.registration_date
    FROM users u
    JOIN users_polars nu ON u.age < nu.age
    """
    )
    print("Join Result:")
    for join_row in join_results.fetchall():
        print(join_row)
    return join_results, join_row


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""# [7. Aggregate Functions](https://duckdb.org/docs/stable/sql/functions/aggregates.html)"""
    )
    return


@app.cell
def _aggregate_functions(database_connection):
    aggregate_results = database_connection.execute(
        """
    SELECT AVG(age) as avg_age, MAX(age) as max_age, MIN(age) as min_age
    FROM (SELECT * FROM users UNION ALL SELECT * FROM users_polars) AS all_users
    """
    ).fetchall()
    print(
        f"Average Age: {aggregate_results[0][0]:.1f}, "
        f"Max Age: {aggregate_results[0][1]}, "
        f"Min Age: {aggregate_results[0][2]}"
    )
    return (aggregate_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""# [8. Converting Results to Polars DataFrames](https://duckdb.org/docs/stable/guides/python/polars.html)"""
    )
    return


@app.cell
def _convert_to_polars(database_connection):
    # -- 8. Converting Results to Polars DataFrames --
    # Convert the result to a Polars DataFrame
    polars_result_df = database_connection.execute("SELECT * FROM users").df()
    print("Result as Polars DataFrame:")
    print(polars_result_df)
    return (polars_result_df,)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import duckdb
    import polars as pl

    return duckdb, mo, pl


if __name__ == "__main__":
    app.run()

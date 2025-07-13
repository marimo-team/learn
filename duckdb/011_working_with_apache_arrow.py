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
        - Combining data from multiple sources
        - Performance benefits
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Convert between DuckDB, Arrow, and Polars/Pandas DataFrames.

        The real power of DuckDB's Arrow integration comes from its seamless interoperability with data frame libraries like Polars and Pandas. Because they all share the Arrow in-memory format, conversions are often zero-copy and extremely fast.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"### From DuckDB to Polars/Pandas")
    return


@app.cell
def _(pl, users_arrow_table):
    # Convert the Arrow table to a Polars DataFrame
    users_polars_df = pl.from_arrow(users_arrow_table)
    users_polars_df
    return (users_polars_df,)


@app.cell
def _(users_arrow_table):
    # Convert the Arrow table to a Pandas DataFrame
    users_pandas_df = users_arrow_table.to_pandas()
    users_pandas_df
    return (users_pandas_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"### From Polars/Pandas to DuckDB")
    return


@app.cell
def _(pl):
    # Create a Polars DataFrame
    polars_df = pl.DataFrame({
        "product_id": [101, 102, 103],
        "product_name": ["Laptop", "Mouse", "Keyboard"],
        "price": [1200.00, 25.50, 75.00]
    })
    polars_df
    return (polars_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"Now we can query this Polars DataFrame directly in DuckDB:")
    return


@app.cell
def _(mo, polars_df):
    # Query the Polars DataFrame directly in DuckDB
    mo.sql(
        f"""
        SELECT product_name, price
        FROM polars_df
        WHERE price > 50
        ORDER BY price DESC;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"Similarly, we can query a Pandas DataFrame:")
    return


@app.cell
def _(pd):
    # Create a Pandas DataFrame
    pandas_df = pd.DataFrame({
        "order_id": [1001, 1002, 1003, 1004],
        "product_id": [101, 102, 103, 101],
        "quantity": [1, 2, 1, 3],
        "order_date": pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-16', '2024-01-17'])
    })
    pandas_df
    return (pandas_df,)


@app.cell
def _(mo, pandas_df):
    # Query the Pandas DataFrame in DuckDB
    mo.sql(
        f"""
        SELECT order_date, SUM(quantity) as total_quantity
        FROM pandas_df
        GROUP BY order_date
        ORDER BY order_date;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Advanced Example: Combining Multiple Data Sources

        One of the most powerful features is the ability to join data from different sources (DuckDB tables, Arrow tables, Polars/Pandas DataFrames) in a single query:
        """
    )
    return


@app.cell
def _(mo, pandas_df, polars_df):
    # Join the DuckDB users table with the Polars products DataFrame and Pandas orders DataFrame
    result = mo.sql(
        f"""
        SELECT 
            u.name as customer_name,
            p.product_name,
            o.quantity,
            p.price,
            (o.quantity * p.price) as total_amount
        FROM users u
        CROSS JOIN pandas_df o
        JOIN polars_df p ON o.product_id = p.product_id
        WHERE u.id = 1  -- Just for Alice
        ORDER BY o.order_date;
        """
    )
    result
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Performance Benefits

        The Arrow format provides several performance benefits:
        
        - **Zero-copy data sharing**: Data can be shared between DuckDB and other Arrow-compatible systems without copying.
        - **Columnar format**: Efficient for analytical queries that typically access a subset of columns.
        - **Type safety**: Arrow's rich type system ensures data types are preserved across systems.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"Let's create a larger dataset to demonstrate the performance:")
    return


@app.cell
def _(pl):
    import time
    
    # Create a larger Polars DataFrame
    large_polars_df = pl.DataFrame({
        "id": range(1_000_000),
        "value": pl.Series([i * 2.5 for i in range(1_000_000)]),
        "category": pl.Series([f"cat_{i % 100}" for i in range(1_000_000)])
    })
    
    print(f"Created DataFrame with {len(large_polars_df):,} rows")
    return large_polars_df, time


@app.cell
def _(large_polars_df, mo, time):
    # Time a query on the large DataFrame
    start_time = time.time()

    result_large = mo.sql(
        f"""
        SELECT 
            category,
            COUNT(*) as count,
            AVG(value) as avg_value,
            MIN(value) as min_value,
            MAX(value) as max_value
        FROM large_polars_df
        GROUP BY category
        ORDER BY count DESC
        LIMIT 10;
        """
    )

    query_time = time.time() - start_time
    print(f"Query completed in {query_time:.3f} seconds")
    
    result_large
    return query_time, result_large, start_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        In this notebook, we've explored:

        1. **Creating Arrow tables from DuckDB queries** using `.to_arrow()`
        2. **Loading Arrow tables into DuckDB** and querying them directly
        3. **Converting between DuckDB, Arrow, Polars, and Pandas** with zero-copy operations
        4. **Combining data from multiple sources** in a single SQL query
        5. **Performance benefits** of using Arrow's columnar format

        The seamless integration between DuckDB and Arrow-compatible systems makes it easy to work with data across different tools while maintaining high performance.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pyarrow as pa
    import polars as pl
    import pandas as pd
    return mo, pa, pd, pl


if __name__ == "__main__":
    app.run()
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "duckdb==1.3.2",
#     "pyarrow==19.0.1",
#     "polars[pyarrow]==1.25.2",
#     "pandas==2.2.3",
#     "sqlglot==27.0.0",
#     "psutil==7.0.0",
#     "altair",
# ]
# ///

import marimo

__generated_with = "0.14.11"
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
    return (users,)


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
def _(mo, users):
    users_arrow_table = mo.sql(  # type: ignore
        """
        SELECT * FROM users WHERE age > 30;
        """
    ).to_arrow()
    return (users_arrow_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `.arrow()` method returns a `pyarrow.Table` object. We can inspect its schema:""")
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
    mo.md(r"""Now, we can query this Arrow table `new_data` directly from SQL by embedding it in the query.""")
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
    mo.md(r"""### From DuckDB to Polars/Pandas""")
    return


@app.cell
def _(pl, users_arrow_table):
    # Convert the Arrow table to a Polars DataFrame
    users_polars_df = pl.from_arrow(users_arrow_table)
    users_polars_df
    return


@app.cell
def _(users_arrow_table):
    # Convert the Arrow table to a Pandas DataFrame
    users_pandas_df = users_arrow_table.to_pandas()
    users_pandas_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### From Polars/Pandas to DuckDB""")
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
    mo.md(r"""Now we can query this Polars DataFrame directly in DuckDB:""")
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
    mo.md(r"""Similarly, we can query a Pandas DataFrame:""")
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
def _(mo, pandas_df, polars_df, users):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Performance Benefits of Arrow Integration

        The zero-copy integration between DuckDB and Apache Arrow delivers significant performance and memory benefits. This seamless integration enables:

        ### Key Benefits:

        - **Memory Efficiency**: Arrow's columnar format uses 20-40% less memory than traditional DataFrames through compact columnar representation and better compression ratios
        - **Zero-Copy Operations**: Data can be shared between DuckDB and Arrow-compatible systems (Polars, Pandas) without any data copying, eliminating redundant memory usage 
        - **Query Performance**: 2-10x faster queries compared to traditional approaches that require data copying
        - **Larger-than-Memory Analysis**: Since both libraries support streaming query results, you can execute queries on data bigger than available memory by processing one batch at a time 
        - **Advanced Query Optimization**: DuckDB's optimizer can push down filters and projections directly into Arrow scans, reading only relevant columns and partitions 
        Let's demonstrate these benefits with concrete examples:
        """
    )
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Memory Efficiency Demonstration""")
    return


@app.cell
def _(pd, pl):
    import sys
    import time

    # Create identical datasets in different formats
    n_rows = 1_000_000

    # Pandas DataFrame (traditional approach)
    pandas_data = pd.DataFrame({
        "id": range(n_rows),
        "value": [i * 2.5 for i in range(n_rows)],
        "category": [f"cat_{i % 100}" for i in range(n_rows)],
        "description": [f"This is a longer text description for row {i}" for i in range(n_rows)]
    })

    # Polars DataFrame (Arrow-based)
    polars_data = pl.DataFrame({
        "id": range(n_rows),
        "value": pl.Series([i * 2.5 for i in range(n_rows)]),
        "category": pl.Series([f"cat_{i % 100}" for i in range(n_rows)]),
        "description": pl.Series([f"This is a longer text description for row {i}" for i in range(n_rows)])
    })

    # Get memory usage
    pandas_memory = pandas_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    polars_memory = polars_data.estimated_size() / 1024 / 1024  # MB

    print(f"Dataset size: {n_rows:,} rows")
    print(f"Pandas memory usage: {pandas_memory:.2f} MB")
    print(f"Polars (Arrow) memory usage: {polars_memory:.2f} MB")
    print(f"Memory savings: {((pandas_memory - polars_memory) / pandas_memory * 100):.1f}%")
    return pandas_data, polars_data, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Performance Comparison: Arrow vs Non-Arrow Approaches""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's compare three approaches for the same analytical query:""")
    return


@app.cell
def _(duckdb, mo, pandas_data, polars_data, time):
    # Test query: group by category and calculate aggregations
    query = """
    SELECT 
        category,
        COUNT(*) as count,
        AVG(value) as avg_value,
        MIN(value) as min_value,
        MAX(value) as max_value,
        SUM(value) as sum_value
    FROM data_source
    GROUP BY category
    ORDER BY count DESC
    """

    # Approach 1: Traditional - Copy data to DuckDB table
    start_time = time.time()
    conn = duckdb.connect(':memory:')
    conn.execute("CREATE TABLE pandas_table AS SELECT * FROM pandas_data")
    result1 = conn.execute(query.replace("data_source", "pandas_table")).fetchall()
    # conn.close()
    approach1_time = time.time() - start_time

    # Approach 2: Direct Pandas query (no DuckDB)
    start_time = time.time()
    result2 = pandas_data.groupby('category').agg({
        'id': 'count',
        'value': ['mean', 'min', 'max', 'sum']
    }).sort_values(('id', 'count'), ascending=False)
    approach2_time = time.time() - start_time

    # Approach 3: Arrow-based - Zero-copy with Polars
    start_time = time.time()
    result3 = mo.sql(
        f"""
        SELECT 
            category,
            COUNT(*) as count,
            AVG(value) as avg_value,
            MIN(value) as min_value,
            MAX(value) as max_value,
            SUM(value) as sum_value
        FROM polars_data
        GROUP BY category
        ORDER BY count DESC
        """
    )
    approach3_time = time.time() - start_time

    print("Performance Comparison:")
    print(f"1. Traditional (copy to DuckDB): {approach1_time:.3f} seconds")
    print(f"2. Pandas groupby: {approach2_time:.3f} seconds")
    print(f"3. Arrow-based (zero-copy): {approach3_time:.3f} seconds")
    print(f"\nSpeedup vs traditional: {approach1_time/approach3_time:.1f}x")
    print(f"Speedup vs pandas: {approach2_time/approach3_time:.1f}x")

    # Return timing variables but not the closed connection
    return approach1_time, approach2_time, approach3_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visualizing the Performance Difference""")
    return


@app.cell
def _(approach1_time, approach2_time, approach3_time, mo, pl):
    import altair as alt
    
    # Create a bar chart showing the performance comparison
    performance_data = pl.DataFrame({
        "Approach": ["Traditional\n(Copy to DuckDB)", "Pandas\nGroupBy", "Arrow-based\n(Zero-copy)"],
        "Time (seconds)": [approach1_time, approach2_time, approach3_time]
    })

    # Create the Altair chart
    chart = alt.Chart(performance_data.to_pandas()).mark_bar().encode(
        x=alt.X("Approach", type="nominal", sort="-y"),
        y=alt.Y("Time (seconds)", type="quantitative"),
        color=alt.Color("Approach", type="nominal", 
                       scale=alt.Scale(range=["#ff6b6b", "#ffd93d", "#6bcf7f"]))
    ).properties(
        title="Query Performance Comparison",
        width=400,
        height=300
    )
    
    # Display using marimo's altair_chart UI element
    mo.ui.altair_chart(chart)
    return alt, chart, performance_data



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Complex Query Performance""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's test a more complex query with joins and window functions:""")
    return


@app.cell
def _(mo, pl, polars_data, time):
    # Create additional datasets for join operations
    categories_df = pl.DataFrame({
        "category": [f"cat_{i}" for i in range(100)],
        "category_group": [f"group_{i // 10}" for i in range(100)],
        "priority": [i % 5 + 1 for i in range(100)]
    })

    # Complex query with join and window functions
    new_start_time = time.time()

    complex_result = mo.sql(
        f"""
        WITH ranked_data AS (
            SELECT 
                d.*,
                c.category_group,
                c.priority,
                ROW_NUMBER() OVER (PARTITION BY c.category_group ORDER BY d.value DESC) as rank_in_group,
                AVG(d.value) OVER (PARTITION BY c.category_group) as group_avg_value
            FROM polars_data d
            JOIN categories_df c ON d.category = c.category
        )
        SELECT 
            category_group,
            COUNT(DISTINCT category) as unique_categories,
            AVG(value) as avg_value,
            MAX(value) as max_value,
            AVG(group_avg_value) as avg_group_value,
            COUNT(CASE WHEN rank_in_group <= 10 THEN 1 END) as top_10_count
        FROM ranked_data
        GROUP BY category_group
        ORDER BY avg_value DESC
        """
    )

    complex_query_time = time.time() - new_start_time
    print(f"Complex query with joins and window functions completed in {complex_query_time:.3f} seconds")

    complex_result
    return (categories_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Memory Efficiency During Operations

    Let's demonstrate how Arrow's zero-copy operations save memory during data transformations:
    """
    )
    return


@app.cell
def _(polars_data, time):
    import psutil
    import os
    import pyarrow.compute as pc  # Add this import

    # Get current process
    process = psutil.Process(os.getpid())

    # Measure memory before operations
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # Perform multiple Arrow-based operations (zero-copy)
    latest_start_time = time.time()

    # These operations use Arrow's zero-copy capabilities
    arrow_table = polars_data.to_arrow()
    arrow_sliced = arrow_table.slice(0, 100000)
    # Use PyArrow compute functions for filtering
    arrow_filtered = arrow_table.filter(pc.greater(arrow_table['value'], 500000))

    arrow_ops_time = time.time() - latest_start_time
    memory_after_arrow = process.memory_info().rss / 1024 / 1024  # MB

    # Compare with traditional copy-based operations
    latest_start_time = time.time()

    # These operations create copies
    pandas_copy = polars_data.to_pandas()
    pandas_sliced = pandas_copy.iloc[:100000].copy()
    pandas_filtered = pandas_copy[pandas_copy['value'] > 500000].copy()

    copy_ops_time = time.time() - latest_start_time
    memory_after_copy = process.memory_info().rss / 1024 / 1024  # MB

    print("Memory Usage Comparison:")
    print(f"Initial memory: {memory_before:.2f} MB")
    print(f"After Arrow operations: {memory_after_arrow:.2f} MB (diff: +{memory_after_arrow - memory_before:.2f} MB)")
    print(f"After copy operations: {memory_after_copy:.2f} MB (diff: +{memory_after_copy - memory_before:.2f} MB)")
    print(f"\nTime comparison:")
    print(f"Arrow operations: {arrow_ops_time:.3f} seconds")
    print(f"Copy operations: {copy_ops_time:.3f} seconds")
    print(f"Speedup: {copy_ops_time/arrow_ops_time:.1f}x")
    return pc



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
    5. **Performance and memory benefits** including:
        - **Memory efficiency**: Arrow format uses 20-40% less memory than traditional DataFrames
        - **Query performance**: 2-10x faster queries through zero-copy operations
        - **Reduced memory overhead**: Operations on Arrow data avoid creating copies
        - **Better scalability**: Can handle larger datasets within the same memory constraints

    The seamless integration between DuckDB and Arrow-compatible systems makes it easy to work with data across different tools while maintaining high performance and memory efficiency.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pyarrow as pa
    import polars as pl
    import pandas as pd
    import duckdb
    import sqlglot
    return duckdb, mo, pa, pd, pl


if __name__ == "__main__":
    app.run()

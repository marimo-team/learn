# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "duckdb==1.3.2",
#     "polars==1.27.0",
#     "numpy==2.2.4",
#     "pyarrow==19.0.1",
#     "pandas==2.2.3",
#     "sqlglot==26.12.1",
#     "plotly==5.23.1",
#     "statsmodels==0.14.4",
# ]
# ///

import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    <p align="center">
      <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxHAqB0W_61zuIGVMiU6sEeQyTaw-9xwiprw&s" alt="DuckDB Image"/>
    </p>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    # 🦆 **DuckDB**: An Embeddable Analytical Database System

    ## What is DuckDB?

    [DuckDB](https://duckdb.org/) is a _high-performance_, in-process, embeddable SQL OLAP (Online Analytical Processing) Database Management System (DBMS) designed for simplicity and speed. It's essentially a fully-featured database that runs directly within your application's process, without needing a separate server. This makes it excellent for complex analytical workloads, offering a robust SQL interface and efficient processing – perfect for learning about databases and data analysis concepts.  It's a great alternative to heavier database systems like PostgreSQL or MySQL when you don't need a full-blown server.

    ---

    ## ⚡ Key Features

    | Feature | Description |
    |:---------|:-------------|
    | **In-Process Architecture** | Runs directly within your application's memory space - no separate server needed, simplifying deployment |
    | **Columnar Storage** | Data stored in columns instead of rows, dramatically improving performance for analytical queries |
    | **Vectorized Execution** | Performs operations on entire columns at once, significantly speeding up data processing |
    | **ACID Transactions** | Ensures data integrity and reliability across operations |
    | **Multi-Language Support** | Provides APIs for `Python`, `R`, `Java`, `C++`, and more |
    | **Zero External Dependencies** | Minimal dependencies, making setup and deployment straightforward |
    | **High Portability** | Works across various operating systems (Windows, macOS, Linux) and hardware architectures |

    ---

    ## [Use Cases](https://github.com/davidgasquez/awesome-duckdb?tab=readme-ov-file):

    - **Data Analysis and Exploration:**  DuckDB is ideal for quickly querying and analyzing datasets, especially for initial exploratory analysis.
    - **Embedded Analytics in Applications:**  You can integrate DuckDB directly into your applications to provide analytical capabilities without the need for a separate database server.
    - **ETL (Extract, Transform, Load) Processes:** DuckDB can be used to perform initial data transformation and cleaning steps as part of an ETL pipeline.
    - **Data Science and Machine Learning Workflows:**  It's a lightweight alternative to larger databases for prototyping data analysis and machine learning models.
    - **Rapid Prototyping of Data Analysis Pipelines:** Quickly test and iterate on data analysis ideas without the complexity of setting up a full-blown database environment.
    - **Small to Medium Datasets:** DuckDB shines when working with datasets that don't require the massive scalability of a traditional database server.

    ---

    ### [Installation](https://duckdb.org/docs/installation/?version=stable&environment=python):

    - Python installation:
    ```
    pip install duckdb
    ```
    ```
    conda install python-duckdb -c conda-forge.
    ```

    <!-- >**_Note_:** DuckDB requires Python 3.7 or newer. You also need to have Python and `pip` or `conda` installed on your system. -->

    /// attention | Note
    DuckDB requires Python 3.7 or newer. You also need to have Python and `pip` or `conda` installed on your system.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # [1. DuckDB Connections: In-Memory vs. File-based](https://duckdb.org/docs/stable/connect/overview.html)

    DuckDB is a lightweight, _relational database management system (RDBMS)_ designed for analytical workloads. Unlike traditional client-server databases, it operates _in-process_ (embedded within your application) and supports both _in-memory_ (temporary) and _file-based_ (persistent) storage.

    ---

    | Feature | In-Memory Connection | File-Based Connection |
    |:---------|:---------------------|:----------------------|
    | Persistence | Temporary (lost when session ends) | Stored on disk (persists between sessions) |
    | Use Cases | Quick analysis, ephemeral data, testing | Long-term storage, data that needs to be accessed later |
    | Performance | Faster for most operations | Slightly slower but provides persistence |
    | Creation | duckdb.connect(':memory:') | duckdb.connect('filename.db') |
    | Multiple Connection Access | Limited to single connection | Multiple connections can access the same database |
    """
    )
    return


@app.cell(hide_code=True)
def _(os):
    # Remove previous database if it exists
    if os.path.exists("example.db"):
        os.remove("example.db")

    if not os.path.exists("data"):
        os.makedirs("data")
    return


@app.cell(hide_code=True)
def _(mo):
    _df = mo.sql(
        f"""
        -- Print the DuckDB version
        SELECT version() AS version_info
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Creating DuckDB Connections

    Let's create both types of DuckDB connections and explore their characteristics.

    1. **In-memory connection**: Data exists only during the current session
    2. **File-based connection**: Data persists between sessions

    We'll then demonstrate the key differences between these connection types.
    """
    )
    return


@app.cell(hide_code=True)
def _(duckdb):
    # Create an in-memory DuckDB connection
    memory_db = duckdb.connect(":memory:")

    # Create a file-based DuckDB connection
    file_db = duckdb.connect("example.db")
    return file_db, memory_db


@app.cell(hide_code=True)
def _(file_db, memory_db):
    # Test both connections
    memory_db.execute(
        "CREATE TABLE IF NOT EXISTS mem_test (id INTEGER, name VARCHAR)"
    )
    memory_db.execute("INSERT INTO mem_test VALUES (1, 'Memory Test')")

    file_db.execute(
        "CREATE TABLE IF NOT EXISTS file_test (id INTEGER, name VARCHAR)"
    )
    file_db.execute("INSERT INTO file_test VALUES (1, 'File Test')")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Testing Connection Persistence

    Let's demonstrate how in-memory databases are ephemeral, while file-based databases persist. 

    1. First, we'll query our tables to confirm the data was properly inserted
    2. Then, we'll simulate an application restart by creating new connections
    3. Finally, we'll check which data persists after the "restart"
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Current Database Contents""")
    return


@app.cell(hide_code=True)
def _(mem_test, memory_db, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM mem_test
        """,
        engine=memory_db
    )
    return


@app.cell(hide_code=True)
def _(file_db, file_test, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM file_test
        """,
        engine=file_db
    )
    return


@app.cell
def _():
    # We don't actually close the connections here since we need them for later cells
    # Just a placeholder for the concept
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""## 🔄 Simulating Application Restart...""")
    return


@app.cell(hide_code=True)
def _(duckdb):
    # Create new connections (simulating restart)
    new_memory_db = duckdb.connect(":memory:")
    new_file_db = duckdb.connect("example.db")
    return new_file_db, new_memory_db


@app.cell(hide_code=True)
def _(new_memory_db):
    # Try to query tables in the new memory connection
    try:
        new_memory_db.execute("SELECT * FROM mem_test").df()
        memory_persistence = "✅ Data persisted in memory (unexpected)"
        memory_data_available = True
    except Exception as e:
        memory_persistence = "❌ Data lost from memory (expected behavior)"
        memory_data_available = False
    return memory_data_available, memory_persistence


@app.cell(hide_code=True)
def _(new_file_db):
    # Try to query tables in the new file connection
    try:
        file_data = new_file_db.execute("SELECT * FROM file_test").df()
        file_persistence = "✅ Data persisted in file (expected behavior)"
        file_data_available = True
    except Exception as e:
        file_persistence = "❌ Data lost from file (unexpected)"
        file_data_available = False
        file_data = None
    return file_data, file_data_available, file_persistence


@app.cell(hide_code=True)
def _(
    file_data_available,
    file_persistence,
    memory_data_available,
    memory_persistence,
    mo,
):
    # Create an interactive display to show persistence results
    persistence_results = mo.ui.table(
        {
            "Connection Type": ["In-Memory Database", "File-Based Database"],
            "Persistence Status": [memory_persistence, file_persistence],
            "Data Available After Restart": [
                memory_data_available,
                file_data_available,
            ],
        }
    )
    return (persistence_results,)


@app.cell(hide_code=True)
def _(mo, persistence_results):
    mo.vstack(
        [
            mo.vstack([mo.md(f"""## Persistence Test Results""")], align="center"),
            persistence_results,
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(file_data, file_data_available, mo):
    if file_data_available:
        mo.md("### Persisted File-Based Data:")
        mo.ui.table(file_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # [2. Creating Tables in DuckDB](https://duckdb.org/docs/stable/sql/statements/create_table.html)

    DuckDB supports standard SQL syntax for creating tables. Let's create more complex tables to demonstrate different data types and constraints.

    ## Table Creation Options

    DuckDB supports various table creation options, including:

    - **Basic tables** with column definitions
    - **Temporary tables** that exist only during the session
    - **CREATE OR REPLACE** to recreate tables
    - **Primary keys** and other constraints
    - **Various data types** including INTEGER, VARCHAR, TIMESTAMP, DECIMAL, etc.
    """
    )
    return


@app.cell(hide_code=True)
def _(file_db, new_memory_db):
    # For the memory database
    try:
        new_memory_db.execute("DROP TABLE IF EXISTS users_memory")
    except:
        pass

    # For the file database
    try:
        file_db.execute("DROP TABLE IF EXISTS users_file")
    except:
        pass
    return


@app.cell(hide_code=True)
def _(file_db, new_memory_db):
    # Create advanced users table in memory database with primary key
    new_memory_db.execute("""
    CREATE TABLE users_memory (
        id INTEGER PRIMARY KEY,
        name VARCHAR NOT NULL,
        age INTEGER CHECK (age > 0),
        email VARCHAR UNIQUE,
        registration_date DATE DEFAULT CURRENT_DATE,
        last_login TIMESTAMP,
        account_balance DECIMAL(10,2) DEFAULT 0.00
    )
    """)

    # Create users table in file database
    file_db.execute("""
    CREATE TABLE users_file (
        id INTEGER PRIMARY KEY,
        name VARCHAR NOT NULL,
        age INTEGER CHECK (age > 0),
        email VARCHAR UNIQUE,
        registration_date DATE DEFAULT CURRENT_DATE,
        last_login TIMESTAMP,
        account_balance DECIMAL(10,2) DEFAULT 0.00
    )
    """)
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    # Get table schema information using DuckDB's internal system tables
    memory_schema = new_memory_db.execute("""
        SELECT column_name, data_type, is_nullable 
        FROM information_schema.columns 
        WHERE table_name = 'users_memory'
        ORDER BY ordinal_position
    """).df()
    return (memory_schema,)


@app.cell(hide_code=True)
def _(memory_schema, mo):
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## 🔍 Table Schema Information """)], align="center"
            ),
            mo.ui.table(memory_schema),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # [3. Inserting Data Into Tables](https://duckdb.org/docs/stable/sql/statements/insert)

    DuckDB supports multiple ways to insert data:

    1. **INSERT INTO VALUES**: Insert specific values
    2. **INSERT INTO SELECT**: Insert data from query results
    3. **Parameterized inserts**: Using prepared statements
    4. **Bulk inserts**: For efficient loading of multiple rows

    Let's demonstrate these different insertion methods:
    """
    )
    return


@app.cell(hide_code=True)
def _(date):
    today = date.today()


    # First check if records already exist to avoid duplicate key errors
    def safe_insert(connection, table_name, data):
        """
        Safely insert data into a table by checking for existing IDs first
        """
        # Check which IDs already exist in the table
        existing_ids = (
            connection.execute(f"SELECT id FROM {table_name}")
            .fetchdf()["id"]
            .tolist()
        )

        # Filter out data with IDs that already exist
        new_data = [record for record in data if record[0] not in existing_ids]

        if not new_data:
            print(
                f"No new records to insert into {table_name}. All IDs already exist."
            )
            return 0

        # Prepare the placeholders for the SQL statement
        placeholders = ", ".join(
            ["(" + ", ".join(["?"] * len(new_data[0])) + ")"] * len(new_data)
        )

        # Flatten the list of tuples for parameter binding
        flat_data = [item for sublist in new_data for item in sublist]

        # Perform the insertion
        if flat_data:
            columns = "(id, name, age, email, registration_date, last_login, account_balance)"
            connection.execute(
                f"INSERT INTO {table_name} {columns} VALUES {placeholders}",
                flat_data,
            )
            return len(new_data)
        return 0
    return (safe_insert,)


@app.cell(hide_code=True)
def _():
    # Prepare the data
    user_data = [
        (
            1,
            "Alice",
            25,
            "alice@example.com",
            "2021-01-01",
            "2023-01-15 14:30:00",
            1250.75,
        ),
        (
            2,
            "Bob",
            30,
            "bob@example.com",
            "2021-02-01",
            "2023-02-10 09:15:22",
            750.50,
        ),
        (
            3,
            "Charlie",
            35,
            "charlie@example.com",
            "2021-03-01",
            "2023-03-05 17:45:10",
            3200.25,
        ),
        (
            4,
            "David",
            40,
            "david@example.com",
            "2021-04-01",
            "2023-04-20 10:30:45",
            1800.00,
        ),
        (
            5,
            "Emma",
            45,
            "emma@example.com",
            "2021-05-01",
            "2023-05-12 11:20:30",
            2500.00,
        ),
        (
            6,
            "Frank",
            50,
            "frank@example.com",
            "2021-06-01",
            "2023-06-18 16:10:15",
            900.25,
        ),
    ]
    return (user_data,)


@app.cell(hide_code=True)
def _(file_db, new_memory_db, safe_insert, user_data):
    # Safely insert data into memory database
    safe_insert(new_memory_db, "users_memory", user_data)

    # Safely insert data into file database
    safe_insert(file_db, "users_file", user_data)
    return


@app.cell(hide_code=True)
def _():
    # If you need to add just one record, you can use a similar approach:
    new_user = (
        7,
        "Grace",
        28,
        "grace@example.com",
        "2021-07-01",
        "2023-07-22 13:45:10",
        1675.50,
    )
    return (new_user,)


@app.cell(hide_code=True)
def _(new_memory_db, new_user):
    # Check if the ID exists before inserting
    if not new_memory_db.execute(
        "SELECT id FROM users_memory WHERE id = ?", [new_user[0]]
    ).fetchone():
        new_memory_db.execute(
            """
            INSERT INTO users_memory (id, name, age, email, registration_date, last_login, account_balance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            new_user,
        )
        print(f"Added user {new_user[1]} to users_memory")
    else:
        print(f"User with ID {new_user[0]} already exists in users_memory")
    return


@app.cell(hide_code=True)
def _(file_db, new_user):
    # Do the same for the file database
    if not file_db.execute(
        "SELECT id FROM users_file WHERE id = ?", [new_user[0]]
    ).fetchone():
        file_db.execute(
            """
            INSERT INTO users_file (id, name, age, email, registration_date, last_login, account_balance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            new_user,
        )
        print(f"Added user {new_user[1]} to users_file")
    else:
        print(f"User with ID {new_user[0]} already exists in users_file")
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    # First try to update
    cursor = new_memory_db.execute(
        """
        UPDATE users_memory
        SET name = ?, age = ?, email = ?, 
            registration_date = ?, last_login = ?, account_balance = ?
        WHERE id = ?
        """,
        (
            "Henry",
            33,
            "henry@example.com",
            "2021-08-01",
            "2023-08-05 09:10:15",
            3100.75,
            8,  # ID should be the last parameter
        ),
    )
    return (cursor,)


@app.cell(hide_code=True)
def _(cursor, mo, new_memory_db):
    # If no rows were updated, perform an insert
    if cursor.rowcount == 0:
        new_memory_db.execute(
            """
            INSERT INTO users_memory
            (id, name, age, email, registration_date, last_login, account_balance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                8,
                "Henry",
                33,
                "henry@example.com",
                "2021-08-01",
                "2023-08-05 09:10:15",
                3100.75,
            ),
        )

    mo.md(
        f"""
        Upserted Henry into users_memory.
        """
    )
    return


@app.cell(hide_code=True)
def _(file_db, mo):
    # For DuckDB using ON CONFLICT, we need to specify the conflict target column
    file_db.execute(
        """
        INSERT INTO users_file (id, name, age, email, registration_date, last_login, account_balance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            age = EXCLUDED.age,
            email = EXCLUDED.email,
            registration_date = EXCLUDED.registration_date,
            last_login = EXCLUDED.last_login,
            account_balance = EXCLUDED.account_balance
        """,
        (
            8,
            "Henry",
            33,
            "henry@example.com",
            "2021-08-01",
            "2023-08-05 09:10:15",
            3100.75,
        ),
    )

    mo.md(
        f"""
        Upserted Henry into users_file.
        """
    )
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    # Display memory data using DuckDB's query capabilities
    memory_results = new_memory_db.execute("""
        SELECT 
            id, 
            name, 
            age, 
            email,
            registration_date,
            last_login,
            account_balance
        FROM users_memory
        ORDER BY id
    """).df()
    return (memory_results,)


@app.cell(hide_code=True)
def _(file_db):
    # Display file data with formatting
    file_results = file_db.execute("""
        SELECT 
            id, 
            name, 
            age, 
            email,
            registration_date,
            last_login,
            CAST(account_balance AS DECIMAL(10,2)) AS account_balance
        FROM users_file
        ORDER BY id
    """).df()
    return (file_results,)


@app.cell(hide_code=True)
def _(file_results, memory_results, mo):
    tabs = mo.ui.tabs(
        {
            "In-Memory Database": mo.ui.table(memory_results),
            "File-Based Database": mo.ui.table(file_results),
        }
    )

    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## 📊 Database Contents After Insertion""")],
                align="center",
            ),
            tabs,
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # [4. Using SQL Directly in marimo](https://duckdb.org/docs/stable/sql/query_syntax/select)

    There are multiple ways to leverage DuckDB's SQL capabilities in marimo:

    1. **Direct execution**: Using DuckDB connections to execute SQL
    2. **marimo SQL**: Using marimo's built-in SQL engine
    3. **Interactive queries**: Combining UI elements with SQL execution

    Let's explore these approaches:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.vstack([mo.md(f"""## 🔍 Query with marimo SQL""")], align="center"),
            mo.md(
                "### marimo has its own [built-in SQL engine](https://docs.marimo.io/guides/working_with_data/sql/) that can work with DataFrames."
            ),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(memory_results, mo):
    # Create a SQL selector for users with age threshold
    age_threshold = mo.ui.slider(
        25, 50, value=30, label="Minimum Age", full_width=True, show_value=True
    )


    # Create a function to filter users based on the slider value
    def filtered_users():
        # Use DuckDB directly instead of mo.sql with users param
        filtered_df = memory_results[memory_results["age"] >= age_threshold.value]
        filtered_df = filtered_df.sort_values("age")
        return mo.ui.table(filtered_df)
    return age_threshold, filtered_users


@app.cell(hide_code=True)
def _(age_threshold, filtered_users, mo):
    layout = mo.vstack(
        [
            mo.md("### Select minimum age:"),
            age_threshold,
            mo.md("### Users meeting age criteria:"),
            filtered_users(),
        ],
        gap=2,
        justify="space-between",
    )

    layout
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# [5. Working with Polars and DuckDB](https://duckdb.org/docs/stable/guides/python/polars.html)""")
    return


@app.cell(hide_code=True)
def _(pl):
    # Create a Polars DataFrame
    polars_df = pl.DataFrame(
        {
            "id": [101, 102, 103],
            "name": ["Product A", "Product B", "Product C"],
            "price": [29.99, 49.99, 19.99],
            "category": ["Electronics", "Furniture", "Books"],
        }
    )
    return (polars_df,)


@app.cell(hide_code=True)
def _(mo, polars_df):
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## Original Polars DataFrame:""")], align="center"
            ),
            mo.ui.table(polars_df),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(new_memory_db, polars_df):
    # Register the Polars DataFrame as a DuckDB table in memory connection
    new_memory_db.register("products_polars", polars_df)

    # Query the registered table
    polars_query_result = new_memory_db.execute(
        "SELECT * FROM products_polars WHERE price > 25"
    ).df()
    return (polars_query_result,)


@app.cell(hide_code=True)
def _(mo, polars_query_result):
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## DuckDB Query Result (From Polars Data):""")],
                align="center",
            ),
            mo.ui.table(polars_query_result),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    # Demonstrate a more complex query
    complex_query_result = new_memory_db.execute("""
        SELECT 
            category,
            COUNT(*) as product_count,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price
        FROM products_polars
        GROUP BY category
        ORDER BY avg_price DESC
    """).df()
    return (complex_query_result,)


@app.cell(hide_code=True)
def _(complex_query_result, mo):
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## Aggregated Product Data by Category:""")],
                align="center",
            ),
            mo.ui.table(complex_query_result),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# [6. Advanced Queries: Joins Between Tables](https://duckdb.org/docs/stable/guides/performance/join_operations.html)""")
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    # Create another table to join with
    new_memory_db.execute("""
    CREATE TABLE IF NOT EXISTS departments (
        id INTEGER,
        department_name VARCHAR,
        manager_id INTEGER
    )
    """)
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    new_memory_db.execute("""
    INSERT INTO departments VALUES
    (101, 'Engineering', 1),
    (102, 'Marketing', 2),
    (103, 'Finance', NULL)
    """)
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    # Execute a join query
    join_result = new_memory_db.execute("""
    SELECT 
        u.id, 
        u.name, 
        u.age, 
        d.department_name
    FROM users_memory u
    LEFT JOIN departments d ON u.id = d.manager_id
    ORDER BY u.id
    """).df()
    return (join_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    <!-- Display the join result -->
    ## Join Result (Users and Departments):
    """
    )
    return


@app.cell
def _(join_result, mo):
    mo.ui.table(join_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    <!-- Demonstrate different types of joins -->
    ## Different Types of Joins
    """
    )
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    # Inner join
    inner_join = new_memory_db.execute("""
    SELECT u.id, u.name, d.department_name
    FROM users_memory u
    INNER JOIN departments d ON u.id = d.manager_id
    """).df()

    # Right join
    right_join = new_memory_db.execute("""
    SELECT u.id, u.name, d.department_name
    FROM users_memory u
    RIGHT JOIN departments d ON u.id = d.manager_id
    """).df()

    # Full outer join
    full_join = new_memory_db.execute("""
    SELECT u.id, u.name, d.department_name
    FROM users_memory u
    FULL OUTER JOIN departments d ON u.id = d.manager_id
    """).df()

    # Cross join
    cross_join = new_memory_db.execute("""
    SELECT u.id, u.name, d.department_name
    FROM users_memory u
    CROSS JOIN departments d
    """).df()

    # Self join (Joining user table with itself to find users with the same age)
    self_join = new_memory_db.execute("""
    SELECT u1.id, u1.name, u2.name AS same_age_user
    FROM users_memory u1
    JOIN users_memory u2 ON u1.age = u2.age AND u1.id <> u2.id
    """).df()

    # Semi join (Finding users who are also managers)
    semi_join = new_memory_db.execute("""
    SELECT u.id, u.name, u.age
        FROM users_memory u
        WHERE EXISTS (
            SELECT 1 FROM departments d 
            WHERE u.id = d.manager_id
    )
    """).df()

    # Anti join (Finding users who are not managers)
    anti_join = new_memory_db.execute("""
    SELECT u.id, u.name, u.age
        FROM users_memory u
        WHERE NOT EXISTS (
            SELECT 1 FROM departments d 
            WHERE u.id = d.manager_id
    )
    """).df()
    return (
        anti_join,
        cross_join,
        full_join,
        inner_join,
        right_join,
        self_join,
        semi_join,
    )


@app.cell(hide_code=True)
def _(mo, new_memory_db):
    # Display base table side by side
    users = new_memory_db.execute("SELECT * FROM users_memory").df()
    departments = new_memory_db.execute("SELECT * FROM departments").df()

    base_tables = mo.vstack(
        [
            mo.vstack([mo.md(f"""# Base Tables""")], align="center"),
            mo.ui.tabs(
                {
                    "User Table": mo.ui.table(users),
                    "Departments Table": mo.ui.table(departments),
                }
            ),
        ]
    )
    base_tables
    return


@app.cell(hide_code=True)
def _(
    anti_join,
    cross_join,
    full_join,
    inner_join,
    join_result,
    mo,
    right_join,
    self_join,
    semi_join,
):
    join_description = {
        "Left Join": "Shows all records from the left table and matching records from the right table. Non-matches filled with NULL.",
        "Inner Join": "Shows only the records where there's a match in both tables.",
        "Right Join": "Shows all records from the right table and matching records from the left table. Non-matches filled with NULL.",
        "Full Outer Join": "Shows all records from both tables, with NULL values where there's no match.",
        "Cross Join": "Returns the Cartesian product - all possible combinations of rows from both tables.",
        "Self Join": "Joins a table with itself, used to compare rows within the same table.",
        "Semi Join": "Returns rows from the first table where one or more matches exist in the second table.",
        "Anti Join": "Returns rows from the first table where no matches exist in the second table.",
    }


    join_tabs = mo.ui.tabs(
        {
            "Left Join": mo.ui.table(join_result),
            "Inner Join": mo.ui.table(inner_join),
            "Right Join": mo.ui.table(right_join),
            "Full Outer Join": mo.ui.table(full_join),
            "Cross Join": mo.ui.table(cross_join),
            "Self Join": mo.ui.table(self_join),
            "Semi Join": mo.ui.table(semi_join),
            "Anti Join": mo.ui.table(anti_join),
        }
    )
    return join_description, join_tabs


@app.cell(hide_code=True)
def _(join_description, join_tabs, mo):
    join_display = mo.vstack(
        [
            mo.vstack([mo.md(f"""# SQL Join Operations""")], align="center"),
            mo.md(f"**{join_tabs.value}**: {join_description[join_tabs.value]}"),
            mo.md("## Join Results"),
            join_tabs,
        ],
        gap=2,
        justify="space-between",
    )

    join_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# [7. Aggregate Functions in DuckDB](https://duckdb.org/docs/stable/sql/functions/aggregates.html)""")
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    # Execute an aggregate query
    agg_result = new_memory_db.execute("""
    SELECT 
        AVG(age) as avg_age, 
        MAX(age) as max_age, 
        MIN(age) as min_age,
        COUNT(*) as total_users,
        SUM(account_balance) as total_balance
    FROM users_memory
    """).df()
    return (agg_result,)


@app.cell(hide_code=True)
def _(agg_result, mo):
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## Aggregate Results (All Users):""")], align="center"
            ),
            mo.ui.table(agg_result),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    age_groups = new_memory_db.execute("""
    SELECT 
        CASE 
            WHEN age < 30 THEN 'Under 30'
            WHEN age BETWEEN 30 AND 40 THEN '30 to 40'
            ELSE 'Over 40'
        END as age_group,
        COUNT(*) as count,
        AVG(age) as avg_age,
        AVG(account_balance) as avg_balance
    FROM users_memory
    GROUP BY 1
    ORDER BY 1
    """).df()
    return (age_groups,)


@app.cell(hide_code=True)
def _(age_groups, mo):
    mo.ui.table(age_groups)
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## Aggregate Results (Grouped by Age Range):""")],
                align="center",
            ),
            mo.ui.table(age_groups),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    window_result = new_memory_db.execute("""
    SELECT 
        id,
        name,
        age,
        account_balance,
        RANK() OVER (ORDER BY account_balance DESC) as balance_rank,
        account_balance - AVG(account_balance) OVER () as diff_from_avg,
        account_balance / SUM(account_balance) OVER () * 100 as pct_of_total
    FROM users_memory
    ORDER BY balance_rank
    """).df()
    return (window_result,)


@app.cell(hide_code=True)
def _(mo, window_result):
    mo.vstack(
        [
            mo.vstack([mo.md(f"""## Window Functions Example""")], align="center"),
            mo.ui.table(window_result),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# [8. Converting DuckDB Results to Polars/Pandas](https://duckdb.org/docs/stable/guides/python/polars.html)""")
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    polars_result = new_memory_db.execute(
        """SELECT * FROM users_memory WHERE age > 25 ORDER BY age"""
    ).pl()
    return (polars_result,)


@app.cell(hide_code=True)
def _(mo, polars_result):
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## Query Result as Polars DataFrame:""")],
                align="center",
            ),
            mo.ui.table(polars_result),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(new_memory_db):
    pandas_result = new_memory_db.execute(
        """SELECT * FROM users_memory WHERE age > 25 ORDER BY age"""
    ).fetch_df()
    return (pandas_result,)


@app.cell(hide_code=True)
def _(mo, pandas_result):
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## Same Query Result as Pandas DataFrame:""")],
                align="center",
            ),
            mo.ui.table(pandas_result),
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## Differences in DataFrame Handling""")],
                align="center",
            ),
            mo.vstack(
                [
                    mo.md(
                        f"""## Polars: Filter users over 35 and calculate average balance"""
                    )
                ],
                align="start",
            ),
        ],
        gap=2, justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(mo, pl, polars_result):
    def _():
        polars_filtered = polars_result.filter(pl.col("age") > 35)
        polars_avg = polars_filtered.select(
            pl.col("account_balance").mean().alias("avg_balance")
        )

        layout = mo.vstack(
            [
                mo.md("### Filtered Polars DataFrame (Age > 35):"),
                mo.ui.table(polars_filtered),
                mo.md("### Average Account Balance:"),
                mo.ui.table(polars_avg),
            ],
            gap=2,
        )
        return layout


    _()
    return


@app.cell(hide_code=True)
def _(mo, pandas_result):
    pandas_avg = pandas_result[pandas_result["age"] > 35]["account_balance"].mean()
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## Pandas: Same operation in pandas style""")],
                align="center",
            ),
            mo.vstack(
                [mo.md(f"""### Average balance: {pandas_avg:.2f}""")],
                align="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# 9. Data Visualization with DuckDB and Plotly""")
    return


@app.cell(hide_code=True)
def _(age_groups, mo, new_memory_db, plotly_express):
    # User distribution by age group
    fig1 = plotly_express.bar(
        age_groups,
        x="age_group",
        y="count",
        title="User Distribution by Age Group",
        labels={"count": "Number of Users", "age_group": "Age Group"},
        color="age_group",
        color_discrete_sequence=plotly_express.colors.qualitative.Plotly,
    )
    fig1.update_traces(
        text=age_groups["count"],
        textposition="outside",
    )
    fig1.update_layout(
        height=450,
        margin=dict(t=50, b=50, l=50, r=25),
        hoverlabel=dict(bgcolor="white", font_size=12),
        template="plotly_white",
    )


    # Average balance by age group
    fig2 = plotly_express.bar(
        age_groups,
        x="age_group",
        y="avg_balance",
        title="Average Account Balance by Age Group",
        labels={"avg_balance": "Average Balance ($)", "age_group": "Age Group"},
        color="age_group",
        color_discrete_sequence=plotly_express.colors.qualitative.Plotly,
    )
    fig2.update_traces(
        text=[f"${val:.2f}" for val in age_groups["avg_balance"]],
        textposition="outside",
    )
    fig2.update_layout(
        height=450,
        margin=dict(t=50, b=50, l=50, r=25),
        hoverlabel=dict(bgcolor="white", font_size=12),
        template="plotly_white",
    )


    # Age vs Account Balance scatter plot
    scatter_data = new_memory_db.execute(
        """
        SELECT 
            name, 
            age, 
            account_balance
        FROM users_memory
        ORDER BY age
        """
    ).df()

    fig3 = plotly_express.scatter(
        scatter_data,
        x="age",
        y="account_balance",
        title="Age vs. Account Balance",
        labels={"account_balance": "Account Balance ($)", "age": "Age"},
        color_discrete_sequence=["#FF7F0E"],
        trendline="ols",
        hover_data=["age", "account_balance"],
        size_max=15,
    )
    fig3.update_traces(marker=dict(size=12))
    fig3.update_layout(
        height=450,
        margin=dict(t=50, b=50, l=50, r=25),
        hoverlabel=dict(bgcolor="white", font_size=12),
        template="plotly_white",
    )


    # Distribution of account balances
    balance_data = new_memory_db.execute(
        """
        SELECT 
            name,
            account_balance
        FROM users_memory
        ORDER BY account_balance DESC
        """
    ).df()

    fig4 = plotly_express.pie(
        balance_data,
        names="name",
        values="account_balance",
        title="Distribution of Account Balances",
        labels={"account_balance": "Account Balance ($)", "name": "User"},
        color_discrete_sequence=plotly_express.colors.qualitative.Pastel,
    )
    fig4.update_traces(textinfo="percent+label", textposition="inside")
    fig4.update_layout(
        height=450,
        margin=dict(t=50, b=50, l=50, r=25),
        hoverlabel=dict(bgcolor="white", font_size=12),
        template="plotly_white",
    )


    category_tabs = mo.ui.tabs(
        {
            "Age Group Analysis": mo.vstack(
                [
                    mo.ui.tabs(
                        {
                            "User Distribution": mo.ui.plotly(fig1),
                            "Average Balance": mo.ui.plotly(fig2),
                        }
                    )
                ],
                gap=2,
                justify="space-between",
            ),
            "Financial Analysis": mo.vstack(
                [
                    mo.ui.tabs(
                        {
                            "Age vs Balance": mo.ui.plotly(fig3),
                            "Balance Distribution": mo.ui.plotly(fig4),
                        }
                    )
                ],
                gap=2,
                justify="space-between",
            ),
        },
        lazy=True,
    )

    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"""## Select a visualization category:""")],
                align="start",
            ),
            category_tabs,
        ],
        gap=2,
        justify="space-between",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition |
    ## Database Management Best Practices
    ///

    ### Closing Connections

    It's important to close database connections when you're done with them, especially for file-based connections:

    ```python
    memory_db.close()
    file_db.close()
    ```

    ### Transaction Management

    DuckDB supports transactions, which can be useful for more complex operations:

    ```python
    conn = duckdb.connect('mydb.db')
    conn.begin()  # Start transaction

    try:
        conn.execute("INSERT INTO users VALUES (1, 'Test User')")
        conn.execute("UPDATE balances SET amount = amount - 100 WHERE user_id = 1")
        conn.commit()  # Commit changes
    except:
        conn.rollback()  # Undo changes if error
        raise
    ```

    ### Query Performance

    DuckDB is optimized for analytical queries. For best performance:

    - Use appropriate data types
    - Create indexes for frequently queried columns
    - For large datasets, consider partitioning
    - Use prepared statements for repeated queries
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""## 10. Interactive DuckDB Dashboard with marimo and Plotly""")
    return


@app.cell(hide_code=True)
def _(mo):
    # Create an interactive filter for age range
    min_age = mo.ui.slider(20, 50, value=25, label="Minimum Age")
    max_age = mo.ui.slider(20, 50, value=50, label="Maximum Age")
    return max_age, min_age


@app.cell(hide_code=True)
def _(max_age, min_age, new_memory_db):
    # Create a function to filter data and update visualizations
    def get_filtered_data(min_val=min_age.value, max_val=max_age.value):
        # Get filtered data based on slider values using parameterized query for safety
        return new_memory_db.execute(
            """
            SELECT 
                id, 
                name, 
                age, 
                email,
                account_balance,
                registration_date
            FROM users_memory
            WHERE age >= ? AND age <= ?
            ORDER BY age
            """,
            [min_val, max_val],
        ).df()
    return (get_filtered_data,)


@app.cell(hide_code=True)
def _(get_filtered_data):
    def get_metrics(data=get_filtered_data()):
        return {
            "user count": len(data),
            "avg_balance": data["account_balance"].mean() if len(data) > 0 else 0,
            "total_balance": data["account_balance"].sum() if len(data) > 0 else 0,
        }
    return (get_metrics,)


@app.cell(hide_code=True)
def _(get_metrics, mo):
    def metrics_display(metrics=get_metrics()):
        return mo.hstack(
            [
                mo.vstack(
                    [
                        mo.md("### Selected Users"),
                        mo.md(f"## {metrics['user count']}"),
                    ],
                    align="center",
                ),
                mo.vstack(
                    [
                        mo.md("### Average Balance"),
                        mo.md(f"## ${metrics['avg_balance']:.2f}"),
                    ],
                    align="center",
                ),
                mo.vstack(
                    [
                        mo.md("### Total Balance"),
                        mo.md(f"## ${metrics['total_balance']:.2f}"),
                    ],
                    align="center",
                ),
            ],
            justify="space-between",
            gap=2,
        )
    return (metrics_display,)


@app.cell(hide_code=True)
def _(get_filtered_data, max_age, min_age, mo, plotly_express):
    def create_visualization(
        data=get_filtered_data(), min_val=min_age.value, max_val=max_age.value
    ):
        if len(data) == 0:
            return mo.ui.text("No data available for the selected age range.")

        # Create visualizations for filtered data
        fig1 = plotly_express.bar(
            data,
            x="name",
            y="account_balance",
            title=f"Account Balance by User (Age {min_val} - {max_val})",
            labels={"account_balance": "Account Balance ($)", "name": "User"},
            color="account_balance",
            color_continuous_scale=plotly_express.colors.sequential.Plasma,
            text_auto=".2s",
        )
        fig1.update_layout(
            height=400,
            xaxis_tickangle=-45,
            margin=dict(t=50, b=70, l=50, r=30),
            hoverlabel=dict(bgcolor="white", font_size=12),
            template="plotly_white",
        )
        fig1.update_traces(
            textposition="outside",
        )

        fig2 = plotly_express.histogram(
            data,
            x="age",
            nbins=min(10, len(set(data["age"]))),
            title=f"Age Distribution (Age {min_val} - {max_val})",
            color_discrete_sequence=["#4C78A8"],
            opacity=0.8,
            histnorm="probability density",
        )
        fig2.update_layout(
            height=400,
            margin=dict(t=50, b=70, l=50, r=30),
            bargap=0.1,
            hoverlabel=dict(bgcolor="white", font_size=12),
            template="plotly_white",
        )

        fig3 = plotly_express.scatter(
            data,
            x="age",
            y="account_balance",
            title=f"Age vs. Account Balance (Age {min_val} - {max_val})",
            labels={"account_balance": "Account Balance ($)", "age": "Age"},
            color="age",
            color_continuous_scale="Viridis",
            size_max=25,
            size="account_balance",
            hover_name="name",
        )
        fig3.update_layout(
            height=400,
            margin=dict(t=50, b=70, l=50, r=30),
            hoverlabel=dict(bgcolor="white", font_size=12),
            template="plotly_white",
        )

        return mo.ui.tabs(
            {
                "Account Balance by User": mo.ui.plotly(fig1),
                "Age Distribution": mo.ui.plotly(fig2),
                "Age vs. Account Balance": mo.ui.plotly(fig3),
            }
        )
    return (create_visualization,)


@app.cell(hide_code=True)
def _(
    create_visualization,
    get_filtered_data,
    max_age,
    metrics_display,
    min_age,
    mo,
):
    def dashboard(
        min_val=min_age.value,
        max_val=max_age.value,
        metrics=metrics_display(),
        data=get_filtered_data(),
        visualization=create_visualization(),
    ):
        return mo.vstack(
            [
                mo.md(f"### Interactive Dashboard (Age {min_val} - {max_val})"),
                metrics,
                mo.md("### Data Table"),
                mo.ui.table(data, page_size=5),
                mo.md("### Visualizations"),
                visualization,
            ],
            gap=2,
            justify="space-between",
        )


    dashboard()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    # Summary and Key Takeaways

    In this notebook, we've explored DuckDB, a powerful embedded analytical database system. Here's what we covered:

    1. **Connection types**: We learned the difference between in-memory databases (temporary) and file-based databases (persistent).

    2. **Table creation**: We created tables with various data types, constraints, and primary keys.

    3. **Data insertion**: We demonstrated different ways to insert data, including single inserts and bulk loading.

    4. **SQL queries**: We executed various SQL queries directly and through marimo's UI components.

    5. **Integration with Polars**: We showed how DuckDB can work seamlessly with Polars DataFrames.

    6. **Joins and relationships**: We performed JOIN operations between tables to combine related data.

    7. **Aggregation**: We used aggregate functions to summarize and analyze data.

    8. **Data conversion**: We converted DuckDB results to both Polars and Pandas DataFrames.

    9. **Best practices**: We reviewed best practices for managing DuckDB connections and transactions.

    10. **Visualization**: We created interactive visualizations and dashboards with Plotly and marimo.

    DuckDB is an excellent tool for data analysis, especially for analytical workloads. Its in-process nature makes it fast and easy to use, while its SQL compatibility makes it accessible for anyone familiar with SQL databases.

    ### Next Steps

    - Try loading larger datasets into DuckDB
    - Experiment with more complex queries and window functions
    - Use DuckDB's COPY functionality to import/export data from/to files
    - Create more advanced interactive dashboards with marimo and Plotly
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import os
    from datetime import date
    import plotly.express as plotly_express
    import plotly.graph_objects as plotly_graph_objects
    import numpy as np
    return date, duckdb, mo, os, pl, plotly_express


if __name__ == "__main__":
    app.run()

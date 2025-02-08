import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # What is Polars?

        [Polars](https://pola.rs/) is a blazingly fast, efficient, and user-friendly DataFrame library designed for data manipulation and analysis in Python. Built with performance in mind, Polars leverages the power of Rust under the hood, enabling it to handle large datasets with ease while maintaining a simple and intuitive API. Whether you're working with structured data, performing complex transformations, or analyzing massive datasets, Polars is designed to deliver exceptional speed and memory efficiency, often outperforming other popular DataFrame libraries like Pandas.

        One of the standout features of Polars is its ability to perform operations in a parallelized and vectorized manner, making it ideal for modern data processing tasks. It supports a wide range of data types, advanced query optimizations, and seamless integration with other Python libraries, making it a versatile tool for data scientists, engineers, and analysts. Additionally, Polars provides a lazy API for deferred execution, allowing users to optimize their workflows by chaining operations and executing them in a single pass.

        With its focus on speed, scalability, and ease of use, Polars is quickly becoming a go-to choice for data professionals looking to streamline their data processing pipelines and tackle large-scale data challenges. Whether you're analyzing gigabytes of data or performing real-time computations, Polars empowers you to work faster and smarter.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        # Why Polars?

        Pandas has long been the go-to library for data manipulation and analysis in Python. However, as datasets grow larger and more complex, Pandas often struggles with performance and memory limitations. This is where Polars shines. Polars is a modern, high-performance DataFrame library designed to address the shortcomings of Pandas while providing a user-friendly experience. 

        Below, we’ll explore key reasons why Polars is a better choice in many scenarios, along with examples.

        ## (a) Easier Syntax Similar 

        Polars is designed with a syntax that is very similar to PySpark while being intuitive like SQL. This makes it easier for data professionals to transition to Polars without a steep learning curve. For example:

        **Example: Filtering and Aggregating Data**

        **In Pandas:**
        ```
        import pandas as pd

        df = pd.DataFrame(
            { 
                "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", 
                           "Male", "Female", "Male", "Female"],
                "Age": [13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
                "Height_CM": [150, 170, 146.5, 142, 155, 165, 170.8, 130, 132.5, 162]
            }
        )

        # query: average height of male and female after the age of 15 years

        # step-1: filter
        filtered_df = df[df["Age"] > 15]

        # step-2: groupby and aggregation
        result = filtered_df.groupby("Gender").mean()
        ```

        **In Polars:**
        ```
        import polars as pl

        df = pd.DataFrame(
            { 
                "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", 
                           "Male", "Female", "Male", "Female"],
                "Age": [13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
                "Height_CM": [150, 170, 146.5, 142, 155, 165, 170.8, 130, 132.5, 162]
            }
        )

        # query: average height of male and female after the age of 15 years

        # filter, groupby and aggregation using method chaining
        result = df_pl.filter(pl.col("Age") > 15).group_by("Gender").agg(pl.mean("Height_CM"))
        ```

        Notice how Polars uses a *method-chaining* approach, similar to PySpark, which makes the code more readable and expressive while using a *single line* to design the query.

        Additionally, Polars supports SQL-like operations *natively*, that allows you to write SQL queries directly on polars dataframe:

        ```
        result = df.sql("SELECT Gender, AVG(Height_CM) FROM self WHERE Age > 15 GROUP BY Gender")
        ```

        ## (b) Scalability - Handling Large Datasets in Memory

        Pandas is limited by its single-threaded design and reliance on Python, which makes it inefficient for processing large datasets. Polars, on the other hand, is built in Rust and optimized for parallel processing, enabling it to handle datasets that are orders of magnitude larger.

        **Example: Processing a Large Dataset**
        In Pandas, loading a large dataset (e.g., 10GB) often results in memory errors:

        ```
        # This may fail with large datasets
        df = pd.read_csv("large_dataset.csv")
        ```

        In Polars, the same operation is seamless:

        ```
        df = pl.read_csv("large_dataset.csv")
        ```

        Polars also supports lazy evaluation, which allows you to optimize your workflows by deferring computations until necessary. This is particularly useful for large datasets:

        ```
        df = pl.scan_csv("large_dataset.csv")  # Lazy DataFrame
        result = df.filter(pl.col("A") > 1).groupby("A").agg(pl.sum("B")).collect()  # Execute
        ```

        ## (c) Compatibility with Other Machine Learning Libraries

        Polars integrates seamlessly with popular machine learning libraries like Scikit-learn, PyTorch, and TensorFlow. Its ability to handle large datasets efficiently makes it an excellent choice for preprocessing data before feeding it into ML models.

        **Example: Preprocessing Data for Scikit-learn**

        ```
        import polars as pl
        from sklearn.linear_model import LinearRegression

        # Load and preprocess data
        df = pl.read_csv("data.csv")
        X = df.select(["feature1", "feature2"]).to_numpy()
        y = df.select("target").to_numpy()

        # Train a model
        model = LinearRegression()
        model.fit(X, y)
        ```

        Polars also supports conversion to other formats like NumPy arrays and Pandas DataFrames, ensuring compatibility with virtually any ML library:

        ```
        # Convert to Pandas DataFrame
        pandas_df = df.to_pandas()

        # Convert to NumPy array
        numpy_array = df.to_numpy()
        ```

        **(d) Additional Advantages of Polars**

        - Rich Functionality: Polars supports advanced operations like window functions, joins, and nested data types, making it a versatile tool for data manipulation.

        - Query Optimization: Polars is significantly faster than Pandas due to its parallelized and vectorized operations. Benchmarks often show Polars outperforming Pandas by 10x or more.

        - Memory Efficiency: Polars uses memory more efficiently, reducing the risk of out-of-memory errors.

        - Lazy API: The lazy evaluation API allows for query optimization and deferred execution, which is particularly useful for complex workflows.
        """
    )
    return


@app.cell
def _():
    import pandas as pd

    df = pd.DataFrame(
        { 
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", 
                       "Male", "Female", "Male", "Female"],
            "Age": [13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            "Height_CM": [150, 170, 146.5, 142, 155, 165, 170.8, 130, 132.5, 162]
        }
    )

    # query: average height of male and female after the age of 15 years
    filtered_df = df[df["Age"] > 15]
    result = filtered_df.groupby("Gender").mean()["Height_CM"]
    result
    return df, filtered_df, pd, result


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    df_pl = pl.DataFrame(
        { 
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", 
                       "Male", "Female", "Male", "Female"],
            "Age": [13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            "Height_CM": [150.0, 170.0, 146.5, 142.0, 155.0, 165.0, 170.8, 130.0, 132.5, 162.0]
        }
    )

    # df_pl
    # query: average height of male and female after the age of 15 years
    result_pl = df_pl.filter(pl.col("Age") > 15).group_by("Gender").agg(pl.mean("Height_CM"))
    result_pl
    return df_pl, result_pl


@app.cell
def _(df_pl):
    df_pl.sql("SELECT Gender, AVG(Height_CM) FROM self WHERE Age > 15 GROUP BY Gender")
    return


@app.cell
def _(mo):
    mo.md(
        """
        # 🔖 References

        - [Polars official website](https://pola.rs/)
        - [Polars Vs. Pandas](https://blog.jetbrains.com/pycharm/2024/07/polars-vs-pandas/)
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

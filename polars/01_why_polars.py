# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas==2.2.3",
#     "polars==1.22.0",
# ]
# ///

import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # An introduction to Polars

        This notebook provides a birds-eye overview of [Polars](https://pola.rs/), a fast and user-friendly data manipulation library for Python, and compares it to alternatives like Pandas and PySpark.

        Like Pandas and PySpark, the central data structure in Polars is **the DataFrame**, a tabular data structure consisting of named columns. For example, the next cell constructs a DataFrame that records the gender, age, and height in centimeters for a number of individuals.
        """
    )
    return


@app.cell
def _():
    import polars as pl

    df_pl = pl.DataFrame(
        { 
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female", 
                       "Male", "Female", "Male", "Female"],
            "age": [13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            "height_cm": [150.0, 170.0, 146.5, 142.0, 155.0, 165.0, 170.8, 130.0, 132.5, 162.0]
        }
    )
    df_pl
    return df_pl, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Unlike Python's earliest DataFrame library Pandas, Polars was designed with performance and usability in mind — Polars can scale to large datasets with ease while maintaining a simple and intuitive API. 

        Polars' performance is due to a number of factors, including its implementation and rust and its ability to perform operations in a parallelized and vectorized manner. It supports a wide range of data types, advanced query optimizations, and seamless integration with other Python libraries, making it a versatile tool for data scientists, engineers, and analysts. Additionally, Polars provides a lazy API for deferred execution, allowing users to optimize their workflows by chaining operations and executing them in a single pass.

        With its focus on speed, scalability, and ease of use, Polars is quickly becoming a go-to choice for data professionals looking to streamline their data processing pipelines and tackle large-scale data challenges.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Choosing Polars over Pandas

        In this section we'll give a few reasons why Polars is a better choice than Pandas, along with examples.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Intuitive syntax

        Polars' syntax is similar to PySpark and intuitive like SQL, making heavy use of **method chaining**. This makes it easy for data professionals to transition to Polars, and leads to an API that is more concise and readable than Pandas.

        **Example.** In the next few cells, we contrast the code to perform a basic filter and aggregation of data with Pandas to the code required to accomplish the same task with `Polars`.
        """
    )
    return


@app.cell
def _():
    import pandas as pd

    df_pd = pd.DataFrame(
        { 
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", 
                       "Male", "Female", "Male", "Female"],
            "Age": [13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            "Height_CM": [150.0, 170.0, 146.5, 142.0, 155.0, 165.0, 170.8, 130.0, 132.5, 162.0]
        }
    )

    # query: average height of male and female after the age of 15 years

    # step-1: filter
    filtered_df_pd = df_pd[df_pd["Age"] > 15]

    # step-2: groupby and aggregation
    result_pd = filtered_df_pd.groupby("Gender")["Height_CM"].mean()
    result_pd
    return df_pd, filtered_df_pd, pd, result_pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The same example can be worked out in Polars more concisely, using method chaining. Notice how the Polars code is essentially as readable as English.""")
    return


@app.cell
def _(pl):
    data_pl = pl.DataFrame(
        { 
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", 
                       "Male", "Female", "Male", "Female"],
            "Age": [13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            "Height_CM": [150.0, 170.0, 146.5, 142.0, 155.0, 165.0, 170.8, 130.0, 132.5, 162.0]
        }
    )

    # query: average height of male and female after the age of 15 years

    # filter, groupby and aggregation using method chaining
    result_pl = data_pl.filter(pl.col("Age") > 15).group_by("Gender").agg(pl.mean("Height_CM"))
    result_pl
    return data_pl, result_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Notice how Polars uses a *method-chaining* approach, similar to PySpark, which makes the code more readable and expressive while using a *single line* to design the query.
        Additionally, Polars supports SQL-like operations *natively*, that allows you to write SQL queries directly on polars dataframe:
        """
    )
    return


@app.cell
def _(data_pl):
    result = data_pl.sql("SELECT Gender, AVG(Height_CM) FROM self WHERE Age > 15 GROUP BY Gender")
    result
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### A large collection of built-in APIs

        Polars has a comprehensive API that enables to perform virtually any operation using built-in methods. In contrast, Pandas often requires more complex operations to be handled using the `apply` method with a lambda function. The issue with `apply` is that it processes rows sequentially, looping through the DataFrame one row at a time, which can be inefficient. By leveraging Polars' built-in methods, you can operate on entire columns at once, unlocking the power of **SIMD (Single Instruction, Multiple Data)** parallelism. This approach not only simplifies your code but also significantly improves performance.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Query optimization 📈

        A key factor behind Polars' performance lies in its **evaluation strategy**. While Pandas defaults to **eager execution**, executing operations in the exact order they are written, Polars offers both **eager and lazy execution**. With lazy execution, Polars employs a **query optimizer** that analyzes all required operations and determines the most efficient way to execute them. This optimization can involve reordering operations, eliminating redundant calculations, and more. 

        For example, consider the following expression to calculate the mean of the `Number1` column for categories "A" and "B" in the `Category` column:

        ```python
        (
            df
            .groupby(by="Category").agg(pl.col("Number1").mean())
            .filter(pl.col("Category").is_in(["A", "B"]))
        )
        ```

        If executed eagerly, the `groupby` operation would first be applied to the entire DataFrame, followed by filtering the results by `Category`. However, with **lazy execution**, Polars can optimize this process by first filtering the DataFrame to include only the relevant categories ("A" and "B") and then performing the `groupby` operation on the reduced dataset. This approach minimizes unnecessary computations and significantly improves efficiency.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Scalability — handling large datasets in memory ⬆️

        Pandas is limited by its single-threaded design and reliance on Python, which makes it inefficient for processing large datasets. Polars, on the other hand, is built in Rust and optimized for parallel processing, enabling it to handle datasets that are orders of magnitude larger.

        **Example: Processing a Large Dataset**
        In Pandas, loading a large dataset (e.g., 10GB) often results in memory errors:

        ```python
        # This may fail with large datasets
        df = pd.read_csv("large_dataset.csv")
        ```

        In Polars, the same operation runs quickly, without memory pressure:

        ```python
        df = pl.read_csv("large_dataset.csv")
        ```

        Polars also supports lazy evaluation, which allows you to optimize your workflows by deferring computations until necessary. This is particularly useful for large datasets:

        ```python
        df = pl.scan_csv("large_dataset.csv")  # Lazy DataFrame
        result = df.filter(pl.col("A") > 1).groupby("A").agg(pl.sum("B")).collect()  # Execute
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Compatibility with other machine learning libraries 🤝

        Polars integrates seamlessly with popular machine learning libraries like Scikit-learn, PyTorch, and TensorFlow. Its ability to handle large datasets efficiently makes it an excellent choice for preprocessing data before feeding it into ML models.

        **Example: Preprocessing Data for Scikit-learn**

        ```python
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

        ```python
        # Convert to Pandas DataFrame
        pandas_df = df.to_pandas()

        # Convert to NumPy array
        numpy_array = df.to_numpy()
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Easy to use, with room for power users

        Polars supports advanced operations like

        - **date handling**
        - **window functions**
        - **joins**
        - **nested data types**

        which is making it a versatile tool for data manipulation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Why not PySpark?

        While **PySpark** is versatile tool that has transformed the way big data is handled and processed in Python, its **complex setup process** can be intimidating, especially for beginners. In contrast, **Polars** requires minimal setup and is ready to use right out of the box, making it more accessible for users of all skill levels.

        When deciding between the two, **PySpark** is the preferred choice for processing large datasets distributed across a **multi-node cluster**. However, for computations on a **single-node machine**, **Polars** is an excellent alternative. Remarkably, Polars is capable of handling datasets that exceed the size of the available RAM, making it a powerful tool for efficient data processing even on limited hardware.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🔖 References

        - [Polars official website](https://pola.rs/)
        - [Polars vs. Pandas](https://blog.jetbrains.com/pycharm/2024/07/polars-vs-pandas/)
        """
    )
    return


if __name__ == "__main__":
    app.run()

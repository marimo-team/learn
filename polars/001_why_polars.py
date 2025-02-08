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

        ## (a) Easier & Intuitive Syntax

        Polars is designed with a syntax that is very similar to PySpark while being intuitive like SQL. This makes it easier for data professionals to transition to Polars without a steep learning curve. For example:

        **Example: Filtering and Aggregating Data**

        **In Pandas:**
        ```{python}
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
        ```{python}
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

        ```{python}
        result = df.sql("SELECT Gender, AVG(Height_CM) FROM self WHERE Age > 15 GROUP BY Gender")
        ```

        ## (b) Large Collection of Built-in APIs

        Polars boasts an **extremely expressive API**, enabling you to perform virtually any operation using built-in methods. In contrast, Pandas often requires more complex operations to be handled using the `apply` method with a lambda function. The issue with `apply` is that it processes rows sequentially, looping through the DataFrame one row at a time, which can be inefficient. By leveraging Polars' built-in methods, you can operate on entire columns at once, unlocking the power of **SIMD (Single Instruction, Multiple Data)** parallelism. This approach not only simplifies your code but also significantly enhances performance.

        ## (c) Query Optimization

        A key factor behind Polars' performance lies in its **evaluation strategy**. While Pandas defaults to **eager execution**, executing operations in the exact order they are written, Polars offers both **eager and lazy execution**. With lazy execution, Polars employs a **query optimizer** that analyzes all required operations and determines the most efficient way to execute them. This optimization can involve reordering operations, eliminating redundant calculations, and more. 

        For example, consider the following expression to calculate the mean of the `Number1` column for categories "A" and "B" in the `Category` column:

        ```{python}
        (
            df
            .groupby(by="Category").agg(pl.col("Number1").mean())
            .filter(pl.col("Category").is_in(["A", "B"]))
        )
        ```

        If executed eagerly, the `groupby` operation would first be applied to the entire DataFrame, followed by filtering the results by `Category`. However, with **lazy execution**, Polars can optimize this process by first filtering the DataFrame to include only the relevant categories ("A" and "B") and then performing the `groupby` operation on the reduced dataset. This approach minimizes unnecessary computations and significantly improves efficiency.

        ## (d) Scalability - Handling Large Datasets in Memory

        Pandas is limited by its single-threaded design and reliance on Python, which makes it inefficient for processing large datasets. Polars, on the other hand, is built in Rust and optimized for parallel processing, enabling it to handle datasets that are orders of magnitude larger.

        **Example: Processing a Large Dataset**
        In Pandas, loading a large dataset (e.g., 10GB) often results in memory errors:

        ```{python}
        # This may fail with large datasets
        df = pd.read_csv("large_dataset.csv")
        ```

        In Polars, the same operation is seamless:

        ```{python}
        df = pl.read_csv("large_dataset.csv")
        ```

        Polars also supports lazy evaluation, which allows you to optimize your workflows by deferring computations until necessary. This is particularly useful for large datasets:

        ```{python}
        df = pl.scan_csv("large_dataset.csv")  # Lazy DataFrame
        result = df.filter(pl.col("A") > 1).groupby("A").agg(pl.sum("B")).collect()  # Execute
        ```

        ## (e) Compatibility with Other ML Libraries

        Polars integrates seamlessly with popular machine learning libraries like Scikit-learn, PyTorch, and TensorFlow. Its ability to handle large datasets efficiently makes it an excellent choice for preprocessing data before feeding it into ML models.

        **Example: Preprocessing Data for Scikit-learn**

        ```{python}
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

        ```{python}
        # Convert to Pandas DataFrame
        pandas_df = df.to_pandas()

        # Convert to NumPy array
        numpy_array = df.to_numpy()
        ```

        ## (f) Rich Functionality

        Polars supports advanced operations like **date handling**, **window functions**, **joins**, and **nested data types**, making it a versatile tool for data manipulation.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        # Why Not PySpark?

        While **PySpark** is undoubtedly a versatile tool that has transformed the way big data is handled and processed in Python, its **complex setup process** can be intimidating, especially for beginners. In contrast, **Polars** requires minimal setup and is ready to use right out of the box, making it more accessible for users of all skill levels.

        When deciding between the two, **PySpark** is the preferred choice for processing large datasets distributed across a **multi-node cluster**. However, for computations on a **single-node machine**, **Polars** is an excellent alternative. Remarkably, Polars is capable of handling datasets that exceed the size of the available RAM, making it a powerful tool for efficient data processing even on limited hardware.
        """
    )
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


if __name__ == "__main__":
    app.run()

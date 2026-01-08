# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "daft==0.4.14",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What Makes Daft Special?

    > _By [Péter Ferenc Gyarmati](http://github.com/peter-gy)_.

    Welcome to the course on [Daft](https://www.getdaft.io/), the distributed dataframe library! In this first chapter, we'll explore what Daft is and what makes it a noteworthy tool in the landscape of data processing. We'll look at its core design choices and how they aim to help you work with data more effectively, whether you're a data engineer, data scientist, or analyst.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎯 Introducing Daft: A Unified Data Engine

    Daft is a distributed query engine designed to handle a wide array of data tasks, from data engineering and analytics to powering ML/AI workflows. It provides both a Python DataFrame API, familiar to users of libraries like Pandas, and a SQL interface, allowing you to choose the interaction style that best suits your needs or the task at hand.

    The main goal of Daft is to provide a robust and versatile platform for processing data, whether it's gigabytes on your laptop or petabytes on a cluster.

    Let's go ahead and `pip install daft` to see it in action!
    """)
    return


@app.cell(hide_code=True)
def _(df_with_discount, discount_slider, mo):
    mo.vstack(
        [
            discount_slider,
            df_with_discount.collect(),
        ]
    )
    return


@app.cell
def _(daft, discount_slider):
    # Let's create a very simple Daft DataFrame
    df = daft.from_pydict(
        {
            "id": [1, 2, 3],
            "product_name": ["Laptop", "Mouse", "Keyboard"],
            "price": [1200, 25, 75],
        }
    )

    # Perform a basic operation: calculate a new price after discount
    df_with_discount = df.with_column(
        "discounted_price",
        df["price"] * (1 - discount_slider.value),
    )
    return (df_with_discount,)


@app.cell(hide_code=True)
def _(mo):
    discount_slider = mo.ui.slider(
        start=0.05,
        stop=0.5,
        step=0.05,
        label="Discount Rate:",
        show_value=True,
    )
    return (discount_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🦀 Built with Rust: Performance and Simplicity

    One of Daft's key characteristics is that its core engine is written in Rust. This choice has several implications for users:

    *   **Performance**: [Rust](https://www.rust-lang.org/) is known for its speed and memory efficiency. Unlike systems built on the Java Virtual Machine (JVM), Rust doesn't have a garbage collector that can introduce unpredictable pauses. This often translates to faster execution and more predictable performance.
    *   **Efficient Python Integration**: Daft uses Rust's native Python bindings. This allows Python code (like your DataFrame operations or User-Defined Functions, which we'll cover later) to interact closely with the Rust engine. This can reduce the overhead often seen when bridging Python with JVM-based systems (e.g., PySpark), especially for custom Python logic.
    *   **Simplified Developer Experience**: Rust-based systems typically require less configuration tuning compared to JVM-based systems. You don't need to worry about JVM heap sizes, garbage collection parameters, or managing Java dependencies.

    Daft also leverages [Apache Arrow](https://arrow.apache.org/) for its in-memory data format. This allows for efficient data exchange between Daft's Rust core and Python, often with zero-copy data sharing, further enhancing performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.image(
            src="https://minio.peter.gy/static/assets/marimo/learn/daft/daft-anti-spark-social-club.jpeg",
            alt="Daft Anti Spark Social Club Meme",
            caption="💡 Fun Fact: Creators of Daft are proud members of the 'Anti Spark Social Club'.",
            width=512,
            height=682,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A cornerstone of Daft's design is **lazy execution**. Imagine defining a DataFrame with a trillion rows on your laptop – usually not a great prospect for your device's memory!
    """)
    return


@app.cell
def _(daft):
    trillion_rows_df = (
        daft.range(1_000_000_000_000)
        .with_column("times_2", daft.col("id") * 2)
        .filter(daft.col("id") % 2 == 0)
    )
    trillion_rows_df
    return (trillion_rows_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With Daft, this is perfectly fine. Operations like `with_column` or `filter` don't compute results immediately. Instead, Daft builds a *logical plan* – a blueprint of the transformations you've defined. You can inspect this plan:
    """)
    return


@app.cell(hide_code=True)
def _(mo, trillion_rows_df):
    mo.mermaid(trillion_rows_df.explain(format="mermaid").split("\nSet")[0][11:-3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This plan is only executed (and data materialized) when you explicitly request it (e.g., with `.show()`, `.collect()`, or by writing to a file). Before execution, Daft's optimizer works to make your query run as efficiently as possible. This approach allows you to define complex operations on massive datasets without immediate computational cost or memory overflow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🌐 Scale Your Work: From Laptop to Cluster

    Daft is designed with scalability in mind. As the trillion-row dataframe example above illustrates, you can write your data processing logic using Daft's Python API, and this same code can run:

    *   **Locally**: Utilizing multiple cores on your laptop or a single powerful machine for development or processing moderately sized datasets.
    *   **On a Cluster**: By integrating with [Ray](https://www.ray.io/), a framework for distributed computing. This allows Daft to scale out to process very large datasets across many machines.

    This "write once, scale anywhere" approach means you don't need to significantly refactor your code when moving from local development to large-scale distributed execution. We'll delve into distributed computing with Ray in a later chapter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🖼️ Handling More Than Just Tables: Multimodal Data Support

    Modern datasets often contain more than just numbers and text. They might include images, audio clips, URLs pointing to external files, tensor data from machine learning models, or complex nested structures like JSON.

    Daft is built to accommodate these **multimodal data types** as integral parts of a DataFrame. This means you can have columns containing image data, embeddings, or other complex Python objects, and Daft provides mechanisms to process them. This is particularly useful for ML/AI pipelines and advanced analytics where diverse data sources are common.

    As an example of how Daft simplifies working with such complex data, let's see how we can process image URLs. With just a few lines of Daft code, we can pull open data from the [National Gallery of Art](https://github.com/NationalGalleryOfArt/opendata), then directly fetch, decode, and even resize the images within our DataFrame:
    """)
    return


@app.cell
def _(daft):
    (
        # Fetch open data from the National Gallery of Art
        daft.read_csv(
            "https://github.com/NationalGalleryOfArt/opendata/raw/refs/heads/main/data/published_images.csv"
        )
        # Working only with first 5 rows to reduce latency of image fetching during this example
        .limit(5)
        # Select the object ID and the image thumbnail URL
        .select(
            daft.col("depictstmsobjectid").alias("objectid"),
            daft.col("iiifthumburl")
            # Download the content from the URL (string -> bytes)
            .url.download(on_error="null")
            # Decode the image bytes into an image object (bytes -> image)
            .image.decode()
            .alias("thumbnail"),
        )
        # Use Daft's built-in image resizing function to create smaller thumbnails
        .with_column(
            "thumbnail_resized",
            # Resize the 'thumbnail' image column
            daft.col("thumbnail").image.resize(w=32, h=32),
        )
        # Execute the plan and bring the results into memory
        .collect()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Example inspired by the great post [Exploring Art with TypeScript, Jupyter, Polars, and Observable Plot](https://deno.com/blog/exploring-art-with-typescript-and-jupyter) published on Deno's blog.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In later chapters, we'll explore in more detail how to work with these image objects and other complex types, including applying User-Defined Functions (UDFs) for custom processing. Until then, you can [take a look at a more complex example](https://blog.getdaft.io/p/we-cloned-over-15000-repos-to-find), in which Daft is used to clone over 15,000 GitHub repos to find the best developers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧑‍💻 Designed for Developers: Python and SQL Interfaces

    Daft aims to be developer-friendly by offering flexible ways to interact with your data:

    *   **Pythonic DataFrame API**: If you've used Pandas, Polars or similar libraries, Daft's Python API for DataFrames will feel quite natural. It provides a rich set of methods for data manipulation, transformation, and analysis.
    *   **SQL Interface**: For those who prefer SQL or have existing SQL-based logic, Daft allows you to write queries using SQL syntax. Daft can execute SQL queries directly or even translate SQL expressions into its native expression system.

    This dual-interface approach allows developers to choose the most appropriate tool for their specific task or leverage existing skills.
    """)
    return


@app.cell
def _(daft):
    df_simple = daft.from_pydict(
        {
            "item_code": [101, 102, 103, 104],
            "quantity": [5, 0, 12, 7],
            "region": ["North", "South", "North", "East"],
        }
    )
    return (df_simple,)


@app.cell
def _(df_simple):
    # Pandas-flavored API
    df_simple.where(
        (df_simple["quantity"] > 0) & (df_simple["region"] == "North")
    ).collect()
    return


@app.cell
def _(daft, df_simple):
    # Polars-flavored API
    df_simple.where(
        (daft.col("quantity") > 0) & (daft.col("region") == "North")
    ).collect()
    return


@app.cell
def _(daft):
    # SQL Interface
    daft.sql(
        "SELECT * FROM df_simple WHERE quantity > 0 AND region = 'North'"
    ).collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🟣 Daft's Value Proposition

    So, what makes Daft special? It's the combination of these design choices:

    *   A **Rust-based core engine** provides a solid foundation for performance and memory management.
    *   **Built-in scalability** means your code can transition from local development to distributed clusters (with Ray) with minimal changes.
    *   **Native handling of multimodal data** opens doors for complex ML/AI and analytics tasks that go beyond traditional tabular data.
    *   **Developer-centric Python and SQL APIs** offer flexibility and ease of use.

    These elements combine to make Daft a versatile tool for tackling modern data challenges.

    And this is just scratching the surface. Daft is a growing data engine with an ambitious vision: to unify data engineering, analytics, and ML/AI workflows 🚀.
    """)
    return


@app.cell
def _():
    import daft
    import marimo as mo
    return daft, mo


if __name__ == "__main__":
    app.run()

# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.2.3",
#     "plotly[express]==6.0.0",
#     "polars==1.28.1",
#     "requests==2.32.3",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Polars with Marimo's Dataframe Transformer

    *By [jesshart](https://github.com/jesshart)*

    The goal of this notebook is to explore Marimo's data explore capabilities alonside the power of polars. Feel free to reference the latest about these Marimo features here: https://docs.marimo.io/guides/working_with_data/dataframes/?h=dataframe#transforming-dataframes
    """
    )
    return


@app.cell
def _(requests):
    json_data = requests.get(
        "https://raw.githubusercontent.com/jesshart/fake-datasets/refs/heads/main/orders.json"
    )
    return (json_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Loading Data
    Let's start by loading our data and getting into the `.lazy()` format so our transformations and queries are speedy.

    Read more about `.lazy()` here: https://docs.pola.rs/user-guide/lazy/
    """
    )
    return


@app.cell
def _(json_data, pl):
    demand: pl.LazyFrame = pl.read_json(json_data.content).lazy()
    demand
    return (demand,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Above, you will notice that when you reference the object as a standalone, you get out-of-the-box convenience from `marimo`. You have the `Table` and `Query Plan` options to choose from. 

    - 💡 Try out the `Table` view! You can click the `Preview data` button to get a quick view of your data.
    - 💡 Take a look at the `Query plan`. Learn more about Polar's query plan here: https://docs.pola.rs/user-guide/lazy/query-plan/
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## marimo's Native Dataframe UI

    There are a few ways to leverage marimo's native dataframe UI. One is by doing what we saw above—by referencing a `pl.LazyFrame` directly. You can also try,

    - Reference a `pl.LazyFrame` (we already did this!)
    - Referencing a `pl.DataFrame` and see how it different from its corresponding lazy version
    - Use `mo.ui.table`
    - Use `mo.ui.dataframe`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Reference a `pl.DataFrame`
    Let's reference the same frame as before, but this time as a `pl.DataFrame` by calling `.collect()` on it.
    """
    )
    return


@app.cell
def _(demand: "pl.LazyFrame"):
    demand.collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note how much functionality we have right out-of-the-box. Click on column names to see rich features like sorting, freezing, filtering, searching, and more!

    Notice how `order_quantity` has a green bar chart under it indicating the distribution of values for the field!

    Don't miss the `Download` feature as well which supports downloading in CSV, json, or parquet format!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Use `mo.ui.table`
    The `mo.ui.table` allows you to select rows for use downstream. You can select the rows you want, and then use these as filtered rows downstream.
    """
    )
    return


@app.cell
def _(demand: "pl.LazyFrame", mo):
    demand_table = mo.ui.table(demand, label="Demand Table")
    return (demand_table,)


@app.cell
def _(demand_table):
    demand_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""I like to use this feature to select groupings based on summary statistics so I can quickly explore subsets of categories. Let me show you what I mean.""")
    return


@app.cell
def _(demand: "pl.LazyFrame", pl):
    summary: pl.LazyFrame = demand.group_by("product_family").agg(
        pl.mean("order_quantity").alias("mean"),
        pl.sum("order_quantity").alias("sum"),
        pl.std("order_quantity").alias("std"),
        pl.min("order_quantity").alias("min"),
        pl.max("order_quantity").alias("max"),
        pl.col("order_quantity").null_count().alias("null_count"),
    )
    return (summary,)


@app.cell
def _(mo, summary: "pl.LazyFrame"):
    summary_table = mo.ui.table(summary)
    return (summary_table,)


@app.cell
def _(summary_table):
    summary_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, instead of manually creating a filter for what I want to take a closer look at, I simply select from the ui and do a simple join to get that aggregated level with more detail.

    The following cell uses the output of the `mo.ui.table` selection, selects its unique keys, and uses that to join for the selected subset of the original table.
    """
    )
    return


@app.cell
def _(demand: "pl.LazyFrame", pl, summary_table):
    selection_keys: pl.LazyFrame = (
        summary_table.value.lazy().select("product_family").unique()
    )
    selection: pl.lazyframe = selection_keys.join(
        demand, on="product_family", how="left"
    )
    selection.collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""You can learn more about joins in Polars by checking out my other interactive notebook here: https://marimo.io/p/@jesshart/basic-polars-joins""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Use `mo.ui.dataframe`""")
    return


@app.cell
def _(demand: "pl.LazyFrame", mo):
    demand_cached = demand.collect()
    mo_dataframe = mo.ui.dataframe(demand_cached)
    return demand_cached, mo_dataframe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Below I simply call the object into view. We will play with it in the following cells.""")
    return


@app.cell
def _(mo_dataframe):
    mo_dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""One way to group this data in polars code directly would be to group by product family to get the mean. This is how it is done in polars:""")
    return


@app.cell
def _(demand_cached, pl):
    demand_agg: pl.DataFrame = demand_cached.group_by("product_family").agg(
        pl.mean("order_quantity").name.suffix("_mean")
    )
    demand_agg
    return (demand_agg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
    ## Try Before You Buy

    1. Now try to do the same summary using Marimo's `mo.ui.dataframe` object above. Also, note how your aggregated column is already renamed! Nice touch!
    2. Try (1) again but use select statements first (This is actually better polars practice anyway since it reduces the frame as you move to aggregation.)

    *When you are ready, check the `Python Code` tab at the top of the table to compare your output to the answer below.*
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mean_code = """
    This may seem verbose compared to what I came up with, but quick and dirty outputs like this are really helpful for quickly exploring the data and learning the polars library at the same time.
    ```python
    df_next = df
    df_next = df_next.group_by(
        [pl.col("product_family")], maintain_order=True
    ).agg(
        [
            pl.col("order_date").mean().alias("order_date_mean"),
            pl.col("order_quantity").mean().alias("order_quantity_mean"),
            pl.col("product").mean().alias("product_mean"),
        ]
    )
    ```
    """

    mean_again_code = """
    ```python
    df_next = df
    df_next = df_next.select(["product_family", "order_quantity"])
    df_next = df_next.group_by(
        [pl.col("product_family")], maintain_order=True
    ).agg(
        [
            pl.col("order_date").mean().alias("order_date_mean"),
            pl.col("order_quantity").mean().alias("order_quantity_mean"),
            pl.col("product").mean().alias("product_mean"),
        ]
    )
    ```
    """
    return mean_again_code, mean_code


@app.cell(hide_code=True)
def _(mean_again_code, mean_code, mo):
    mo.accordion(
        {
            "Show Code (1)": mean_code,
            "Show Code (2)": mean_again_code,
        }
    )
    return


@app.cell
def _(demand_agg: "pl.DataFrame", mo, px):
    bar_graph = px.bar(
        demand_agg,
        x="product_family",
        y="order_quantity_mean",
        title="Mean Quantity over Product Family",
    )

    note: str = """
    Note: This graph will only show if the above mo_dataframe is correct!

    If you want more on interactive graphs, check out https://github.com/marimo-team/learn/blob/main/polars/05_reactive_plots.py
    """

    mo.vstack(
        [
            mo.md(note),
            bar_graph,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # About this Notebook
    Polars and Marimo are both relatively new to the data wrangling space, but their power (and the thrill of their use) cannot be overstated—well, I suppose it could, but you get the meaning. In this notebook, you learn how to leverage basic Polars skills to load-in and explore your data in concert with Marimo's powerful UI elements.

    ## 📚 Documentation References

    - **Marimo: Dataframe Transformation Guide**  
      https://docs.marimo.io/guides/working_with_data/dataframes/?h=dataframe#transforming-dataframes

    - **Polars: Lazy API Overview**  
      https://docs.pola.rs/user-guide/lazy/

    - **Polars: Query Plan Explained**  
      https://docs.pola.rs/user-guide/lazy/query-plan/

    - **Marimo Notebook: Basic Polars Joins (by jesshart)**  
      https://marimo.io/p/@jesshart/basic-polars-joins

    - **Marimo Learn: Interactive Graphs with Polars**  
      https://github.com/marimo-team/learn/blob/main/polars/05_reactive_plots.py
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    import requests
    import json
    import plotly.express as px
    return pl, px, requests


if __name__ == "__main__":
    app.run()

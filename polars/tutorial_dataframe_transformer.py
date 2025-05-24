# /// script
# dependencies = [
#     "marimo",
#     "polars==1.28.1",
#     "requests==2.32.3",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import requests
    import json
    return mo, pl, requests


@app.cell
def _(requests):
    json_data = requests.get(
        "https://raw.githubusercontent.com/jesshart/fake-datasets/refs/heads/main/orders.json"
    )
    return (json_data,)


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        r"""
        Above, you will notice that when you reference the object as a standalone, you get out-of-the-box convenince from `marimo`. You have the `Table` and `Query Plan` options to choose from. 

        - 💡 Try out the `Table` view! You can click the `Preview data` button to get a quick view of your data.
        - 💡 Take a look at the `Query plan`. Learn more about Polar's query plan here: https://docs.pola.rs/user-guide/lazy/query-plan/
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # marimo's Native Dataframe UI

        There are a few ways to leverage marimo's native dataframe UI. One is by doing what we saw above—by referencing a `pl.LazyFrame` directly. You can also try,

        - Reference a `pl.LazyFrame` (we already did this!)
        - Referencing a `pl.DataFrame` and see how it different from its corresponding lazy version
        - Use `mo.ui.table`
        - Use `mo.ui.dataframe`
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Reference a pl.DataFrame
        Let's reference the same frame as before, but this time as a `pl.DataFrame` by calling `.collect()` on it.
        """
    )
    return


@app.cell
def _(demand):
    demand.collect()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Note how much functionality we have right out-of-the-box. Click on column names to see rich features like sorting, freezing, filtering, searching, and more!

        Notice how `order_quantity` has a green bar chart under it indicating the ditribution of values for the field!

        Don't miss the `Download` feature as well which supports downloading in CSV, json, or parquet format!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Use `mo.ui.table`
        The `mo.ui.table` allows you to select rows for use downstream. You can select the rows you want, and then use these as filtered rows downstream.
        """
    )
    return


@app.cell
def _(demand, mo):
    demand_table = mo.ui.table(demand, label="Demand Table")
    return (demand_table,)


@app.cell
def _(demand_table):
    demand_table
    return


@app.cell
def _(mo):
    mo.md(r"""I like to use this feature to select groupings based on summary statistics so I can quickly explore subsets of categories. Let me show you what I mean.""")
    return


@app.cell
def _(demand, pl):
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
def _(mo, summary):
    summary_table = mo.ui.table(summary)
    return (summary_table,)


@app.cell
def _(summary_table):
    summary_table
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Now, instead of manually creatinga filter for what I want to take a closer look at, I simply select from the ui and do a simple join to get that aggregated level with more detail.

        The following cell uses the output of the `mo.ui.table` selection, selects its unique keys, and uses that to join for the selected subset of the original table.
        """
    )
    return


@app.cell
def _(demand, pl, summary_table):
    selection_keys: pl.LazyFrame = (
        summary_table.value.lazy().select("product_family").unique()
    )
    selection: pl.lazyframe = selection_keys.join(
        demand, on="product_family", how="left"
    )
    selection.collect()
    return


@app.cell
def _(mo):
    mo.md(r"""## Use `mo.ui.dataframe`""")
    return


@app.cell
def _(demand, mo):
    mo_dateframe = mo.ui.dataframe(demand.collect())
    return (mo_dateframe,)


@app.cell
def _(mo_dateframe):
    mo_dateframe
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

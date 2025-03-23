# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "polars==1.23.0",
# ]
# ///

import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Aggregations
        _By [Joram Mutenge](https://www.udemy.com/user/joram-mutenge/)._

        In this notebook, you'll learn how to perform different types of aggregations in Polars, including grouping by categories and time. We'll analyze sales data from a clothing store, focusing on three product categories: hats, socks, and sweaters.
        """
    )
    return


@app.cell
def _():
    import polars as pl

    df = (pl.read_csv('https://raw.githubusercontent.com/jorammutenge/learn-rust/refs/heads/main/sample_sales.csv', try_parse_dates=True)
          .rename(lambda col: col.replace(' ','_').lower())
         )
    df
    return df, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Grouping by category
        ### With single category
        Let's find out how many of each product category we sold.
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .group_by('category')
     .agg(pl.sum('quantity'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It looks like we sold more sweaters. Maybe this was a winter season.

        Let's add another aggregate to see how much was spent on the total units for each product.
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .group_by('category')
     .agg(pl.sum('quantity'),
          pl.sum('ext_price'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We could also write aggregate code for the two columns as a single line.""")
    return


@app.cell
def _(df, pl):
    (df
     .group_by('category')
     .agg(pl.sum('quantity','ext_price'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Actually, the way we've been writing the aggregate lines is syntactic sugar. Here's a longer way of doing it as shown in the [Polars documentation](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.dataframe.group_by.GroupBy.agg.html).""")
    return


@app.cell
def _(df, pl):
    (df
     .group_by('category')
     .agg(pl.col('quantity').sum(),
          pl.col('ext_price').sum())
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### With multiple categories
        We can also group by multiple categories. Let's find out how many items we sold in each product category for each SKU. This more detailed aggregation will produce more rows than the previous DataFrame.
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .group_by('category','sku')
     .agg(pl.sum('quantity'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Aggregations when grouping data are not limited to sums. You can also use functions like [`max`, `min`, `median`, `first`, and `last`](https://docs.pola.rs/user-guide/expressions/aggregation/#basic-aggregations).  

        Let's find the largest sale quantity for each product category.
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .group_by('category')
     .agg(pl.max('quantity'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's make the aggregation more interesting. We'll identify the first customer to purchase each item, along with the quantity they bought and the amount they spent.

        **Note:** To make this work, we'll have to sort the date from earliest to latest.
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .sort('date')
     .group_by('category')
     .agg(pl.first('account_name','quantity','ext_price'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Grouping by time
        Since `datetime` is a special data type in Polars, we can perform various group-by aggregations on it.  

        Our dataset spans a two-year period. Let's calculate the total dollar sales for each year. We'll do it the naive way first so you can appreciate grouping with time.
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(year=pl.col('date').dt.year())
     .group_by('year')
     .agg(pl.sum('ext_price').round(2))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We had more sales in 2014.

        Now let's perform the above operation by groupin with time. This requires sorting the dataframe first.
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .sort('date')
     .group_by_dynamic('date', every='1y')
     .agg(pl.sum('ext_price'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The beauty of grouping with time is that it allows us to resample the data by selecting whatever time interval we want.

        Let's find out what the quarterly sales were for 2014
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .filter(pl.col('date').dt.year() == 2014)
     .sort('date')
     .group_by_dynamic('date', every='1q')
     .agg(pl.sum('ext_price'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here's an interesting question we can answer that takes advantage of grouping by time.

        Let's find the hour of the day where we had the most sales in dollars.
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .sort('date')
     .group_by_dynamic('date', every='1h')
     .agg(pl.max('ext_price'))
     .filter(pl.col('ext_price') == pl.col('ext_price').max())
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Just for fun, let's find the median number of items sold in each SKU and the total dollar amount in each SKU every six days.""")
    return


@app.cell
def _(df, pl):
    (df
     .sort('date')
     .group_by_dynamic('date', every='6d')
     .agg(pl.first('sku'),
          pl.median('quantity'),
          pl.sum('ext_price'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's rename the columns to clearly indicate the type of aggregation performed. This will help us identify the aggregation method used on a column without needing to check the code.""")
    return


@app.cell
def _(df, pl):
    (df
     .sort('date')
     .group_by_dynamic('date', every='6d')
     .agg(pl.first('sku'),
          pl.median('quantity').alias('median_qty'),
          pl.sum('ext_price').alias('total_dollars'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Grouping with over

        Sometimes, we may want to perform an aggregation but also keep all the columns and rows of the dataframe.

        Let's assign a value to indicate the number of times each customer visited and bought something.
        """
    )
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(buy_freq=pl.col('account_name').len().over('account_name'))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, let's determine which customers visited the store the most and bought something.""")
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(buy_freq=pl.col('account_name').len().over('account_name'))
     .filter(pl.col('buy_freq') == pl.col('buy_freq').max())
     .select('account_name','buy_freq')
     .unique()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""There's more you can do with aggregations in Polars such as [sorting with aggregations](https://docs.pola.rs/user-guide/expressions/aggregation/#sorting). We hope that in this notebook, we've armed you with the tools to get started.""")
    return


if __name__ == "__main__":
    app.run()

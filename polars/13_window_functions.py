# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "duckdb==1.2.2",
#     "marimo",
#     "polars==1.29.0",
#     "pyarrow==20.0.0",
#     "sqlglot==26.16.4",
# ]
# ///

import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium", app_title="Window Functions")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Window Functions
        _By [Henry Harbeck](https://github.com/henryharbeck)._

        In this notebook, you'll learn how to perform different types of window functions in Polars.
        You'll work with partitions, ordering and Polars' available "mapping strategies".

        We'll use a dataset with a few days of paid and organic digital revenue data.
        """
    )
    return


@app.cell
def _():
    from datetime import date

    import polars as pl

    dates = pl.date_range(date(2025, 2, 1), date(2025, 2, 5), eager=True)

    df = pl.DataFrame(
        {
            "date": pl.concat([dates, dates]).sort(),
            "channel": ["Paid", "Organic"] * 5,
            "revenue": [6000, 2000, 5200, 4500, 4200, 5900, 3500, 5000, 4800, 4800],
        }
    )

    df
    return date, dates, df, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What is a window function?

        A window function performs a calculation across a set of rows that are related to the current row.
        They allow you to perform aggregations and other calculations within a group without collapsing
        the number of rows (opposed to a group by aggregation, which does collapse the number of rows). Typically the result of a
        window function is assigned back to rows within the group, but Polars also offers additional alternatives.

        Window functions can be used by specifying the [`over`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.over.html)
        method on an expression.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Partitions
        Partitions are the "group by" columns. We will have one "window" of data per unique value in the partition column(s), to
        which the function will be applied.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Partitioning by a single column

        Let's get the total revenue per date...
        """
    )
    return


@app.cell
def _(df, pl):
    daily_revenue = pl.col("revenue").sum().over("date")

    df.with_columns(daily_revenue.alias("daily_revenue"))
    return (daily_revenue,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And then see what percentage of the daily total was Paid and what percentage was Organic.""")
    return


@app.cell
def _(daily_revenue, df, pl):
    df.with_columns(daily_revenue_pct=(pl.col("revenue") / daily_revenue))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's now calculate the maximum revenue, cumulative revenue, rank the revenue and calculate the day-on-day change,
        all partitioned (split) by channel.
        """
    )
    return


@app.cell
def _(df, pl):
    df.with_columns(
        maximum_revenue=pl.col("revenue").max().over("channel"),
        cumulative_revenue=pl.col("revenue").cum_sum().over("channel"),
        revenue_rank=pl.col("revenue").rank(descending=True).over("channel"),
        day_on_day_change=pl.col("revenue").diff().over("channel"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that aggregation functions such as `sum` and `max` have their value applied back to each row in the partition
        (group). Non-aggregate functions such as `cum_sum`, `rank` and `diff` can produce different values per row, but
        still only consider rows within their partition.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Partitioning by multiple columns

        We can also partition by multiple columns.

        Let's add a column to see whether it is a weekday (business day), then get the maximum revenue by that and
        the channel.
        """
    )
    return


@app.cell
def _(df, pl):
    (
        df.with_columns(
            is_weekday=pl.col("date").dt.is_business_day(),
        ).with_columns(
            max_rev_by_channel_and_weekday=pl.col("revenue").max().over("is_weekday", "channel"),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Partitioning by expressions

        Polars also lets you partition by expressions without needing to create them as columns first.

        So, we could re-write the previous window function as...
        """
    )
    return


@app.cell
def _(df, pl):
    df.with_columns(
        max_rev_by_channel_and_weekday=pl.col("revenue")
        .max()
        .over((pl.col("date").dt.is_business_day()), "channel")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Window functions fit into Polars' composable [expressions API](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/#expressions),
        so can be combined with all [aggregation methods](https://docs.pola.rs/api/python/stable/reference/expressions/aggregation.html)
        and methods that consider more than 1 row (e.g., `cum_sum`, `rank` and `diff` as we just saw).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ordering

        The `order_by` parameter controls how to order the data within the window. The function is applied to the data in this
        order.

        Up until this point, we have been letting Polars do the window function calculations based on the order of the rows in the
        DataFrame. There can be times where we would like order of the calculation and the order of the output itself to differ.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Ordering in a window function

        Let's say we want the DataFrame ordered by day of week, but we still want cumulative revenue and the first revenue observation, both
        ordered by date and partitioned by channel...
        """
    )
    return


@app.cell
def _(df, pl):
    # Monday = 1, Sunday = 7
    df_sorted = (
        df.sort(pl.col("date").dt.weekday())
        # Show the weekday for transparency
        .with_columns(pl.col("date").dt.to_string("%a").alias("weekday"))
    )

    df_sorted.select(
        "date",
        "weekday",
        "channel",
        "revenue",
        pl.col("revenue").cum_sum().over("channel", order_by="date").alias("cumulative_revenue"),
        pl.col("revenue").first().over("channel", order_by="date").alias("first_revenue"),
    )
    return (df_sorted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Note about window function ordering compared to SQL

        It is worth noting that traditionally in SQL, many more functions require an `ORDER BY` within `OVER` than in
        equivalent functions in Polars.

        For example, an SQL `RANK()` expression like...
        """
    )
    return


@app.cell
def _(df, mo):
    _df = mo.sql(
        f"""
        SELECT
            date,
            channel,
            revenue,
            RANK() OVER (PARTITION BY channel ORDER BY revenue DESC) AS revenue_rank
        FROM df
        -- re-sort the output back to the original order for ease of comparison
        ORDER BY date, channel DESC
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ...does not require an `order_by` in Polars as the column and the function are already bound (including with the
        `descending=True` argument).
        """
    )
    return


@app.cell
def _(df, pl):
    df.select(
        "date",
        "channel",
        "revenue",
        revenue_rank=pl.col("revenue").rank(descending=True).over("channel"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Descending order

        We can also order in descending order by passing `descending=True`...
        """
    )
    return


@app.cell
def _(df_sorted, pl):
    (
        df_sorted.select(
            "date",
            "weekday",
            "channel",
            "revenue",
            pl.col("revenue").cum_sum().over("channel", order_by="date").alias("cumulative_revenue"),
            pl.col("revenue").first().over("channel", order_by="date").alias("first_revenue"),
            pl.col("revenue")
            .first()
            .over("channel", order_by="date", descending=True)
            .alias("last_revenue"),
            # Or, alternatively
            pl.col("revenue").last().over("channel", order_by="date").alias("also_last_revenue"),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Mapping Strategies

        Mapping Strategies control how Polars maps the result of the window function back to the original DataFrame

        Generally (by default) the result of a window function is assigned back to rows within the group. Through Polars' mapping
        strategies, we will explore other possibilities.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Group to rows

        "group_to_rows" is the default mapping strategy and assigns the result of the window function back to the rows in the
        window.
        """
    )
    return


@app.cell
def _(df, pl):
    df.with_columns(
        cumulative_revenue=pl.col("revenue").cum_sum().over("channel", mapping_strategy="group_to_rows")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Join

        The "join" mapping strategy aggregates the resulting values in a list and repeats the list for all rows in the group.
        """
    )
    return


@app.cell
def _(df, pl):
    df.with_columns(
        cumulative_revenue=pl.col("revenue").cum_sum().over("channel", mapping_strategy="join")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explode

        The "explode" mapping strategy is similar to "group_to_rows", but is typically faster and does not preserve the order of
        rows. Due to this, it requires sorting columns (including those not in the window function) for the result to make sense.
        It should also only be used in a `select` context and not `with_columns`.

        The result of "explode" is similar to a `group_by` followed by an `agg` followed by an `explode`.
        """
    )
    return


@app.cell
def _(df, pl):
    df.select(
        pl.all().over("channel", order_by="date", mapping_strategy="explode"),
        cumulative_revenue=pl.col("revenue")
        .cum_sum()
        .over("channel", order_by="date", mapping_strategy="explode"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Note the modified order of the rows in the output, (but data is the same)...""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Other tips and tricks""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Reusing a window

        In SQL there is a `WINDOW` keyword, which easily allows the re-use of the same window specification across expressions
        without needing to repeat it. In Polars, this can be achieved by using `dict` unpacking to pass arguments to `over`.
        """
    )
    return


@app.cell
def _(df_sorted, pl):
    window = {
        "partition_by": "date",
        "order_by": "date",
        "mapping_strategy": "group_to_rows",
    }

    df_sorted.with_columns(
        pct_daily_revenue=(pl.col("revenue") / pl.col("revenue").sum()).over(**window),
        highest_revenue_channel=pl.col("channel").top_k_by("revenue", k=1).first().over(**window),
        daily_revenue_rank=pl.col("revenue").rank().over(**window),
        cumulative_daily_revenue=pl.col("revenue").cum_sum().over(**window),
    )
    return (window,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Rolling Windows

        Much like in SQL, Polars also gives you the ability to do rolling window computations. In Polars, the rolling calculation
        is also aware of temporal data, making it easy to express if the data is not contiguous (i.e., observations are missing).

        Let's look at an example of that now by filtering out one day of our data and then calculating both a 3-day and 3-row
        max revenue split by channel...
        """
    )
    return


@app.cell
def _(date, df, pl):
    (
        df.filter(pl.col("date") != date(2025, 2, 2))
        .with_columns(
            # "3d" -> 3 days
            rev_3_day_max=pl.col("revenue").rolling_max_by("date", "3d", min_samples=1).over("channel"),
            rev_3_row_max=pl.col("revenue").rolling_max(3, min_samples=1).over("channel"),
        )
        # sort to make the output a little easier to analyze
        .sort("channel", "date")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Notice the difference in the 2nd last row...""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We hope you enjoyed this notebook, demonstrating window functions in Polars!""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Additional References

        - [Polars User guide - Window functions](https://docs.pola.rs/user-guide/expressions/window-functions/)
        - [Polars over method API reference](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.over.html)
        - [PostgreSQL window function documentation](https://www.postgresql.org/docs/current/tutorial-window.html)
        """
    )
    return


if __name__ == "__main__":
    app.run()

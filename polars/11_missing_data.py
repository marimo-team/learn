# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars==1.22.0",
# ]
# ///

import marimo

__generated_with = "0.11.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Handling Missing Data in Polars

        _By Felix Najera._  

        In real‑world datasets, missing values are inevitable. Polars offers a rich set of methods to detect, remove, and impute missing data efficiently. This notebook walks through common patterns and best practices for dealing with nulls in Polars DataFrames.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "### TLDR":"""You can identify and manage missing values using specific methods: [```.is_null```](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.is_null.html) to check for missing values, [```fill_null```](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.fill_null.html) replaces them with specified values, and [`drop`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.drop.html) removes rows or columns containing null values, applicable to all data types including strings. Taking special care of variables with String cases see the Strings notebook for more."""

    })
    return


@app.cell
def _(mo):
    mo.md("""Polars datatype specific features for missing data are NaN(for float values) and null(for everything else).""")
    return


@app.cell
def _():
    import polars as pl

    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", None, "apple", "apple", "banana"],
            "age": [25, None, 37, 29, None],
            "B": [5, 4, 3, 2, 1],
            "score": [85, 92,  None, None, 88],
            "height_cm": [170.0, 165.5, None, 180.2, 175.0],

        }
    )
    df
    return df, pl


@app.cell
def _(mo):
    mo.md("""Where ```null``` is what is used in dataset features of with datatype other floats and ```NaN``` is used in the dataset features witha  datatype of float""")
    return


@app.cell
def _(mo):
    mo.md("""For the purposes of this guide we won't mention all pelicularities that may come from alternative dataframes found in other packages such as Pandas.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Detecting Nulls

        Use `is_null` or `null_count` to find missing values.
        """
    )
    return


@app.cell
def _(df, pl):
    # Show where nulls are
    mask = df.select([pl.col(c).is_null().alias(f"{c}_is_null") for c in df.columns])
    counts = df.null_count()
    mask, counts
    return counts, mask


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Dropping Nulls

        - `drop_nulls()` removes any row containing nulls  
        - You can target specific columns with `subset=`
        """
    )
    return


@app.cell
def _(df):
    # drop any row with a null
    dropped_all = df.drop_nulls()
    # drop rows with nulls only in 'score'
    dropped_score = df.drop_nulls(subset=["score"])
    dropped_all, dropped_score
    return dropped_all, dropped_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Filling Nulls

        - `fill_null(value)` replaces all nulls with a scalar  
        - `fill_null(strategy="forward")` uses previous non-null value  
        - `fill_null(strategy="backward")` uses next non-null value
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion({"Sidenote":"""Theres also an alternative function ```interpolate``` that can be used to fill missing values in a column with the average of the surrounding values. This is useful when you have a time series or a sequence of values and you want to fill in the gaps with a smooth transition."""})
    return


@app.cell
def _(df):
    filled_constant = df.fill_null(0)
    filled_ffill = df.fill_null(strategy="forward")
    filled_bfill = df.fill_null(strategy="backward")
    filled_constant, filled_ffill, filled_bfill
    return filled_bfill, filled_constant, filled_ffill


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Imputing with Expressions

        Leverage Polars expressions for conditional imputation.  
        Example: fill missing `age` with the average age.
        """
    )
    return


@app.cell
def _(df, pl):
    mean_age = df.select(pl.col("age").mean()).item()
    imputed = df.with_columns(
        pl.col("age").fill_null(mean_age).alias("age_imputed")
    )
    mean_age, imputed
    return imputed, mean_age


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🔖 References

        - Handling missing data in Polars: https://docs.pola.rs/user-guide/expressions/missing-data/
        """
    )
    return


if __name__ == "__main__":
    app.run()

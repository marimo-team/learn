# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars==1.22.0",
# ]
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Handling Missing Data in Polars

    _By Felix Najera (https://github.com/folicks)._  

    In real‑world datasets, missing values are inevitable. Polars offers a rich set of methods to detect, remove, and impute missing data efficiently. This notebook walks through common patterns and best practices for dealing with nulls in Polars DataFrames.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | TLDR
        type: info
    You can identify and manage missing values using specific methods: [```.is_null```](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.is_null.html) to check for missing values, [```fill_null```](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.fill_null.html) replaces them with specified values, and [`drop`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.drop.html) removes rows or columns containing null values, applicable to all data types including strings. Taking special care of variables with type [`String`](https://marimo.app/github.com/marimo-team/learn/blob/main/polars/10_strings.py
    ) see the Strings notebook for more.

    ///
    """
    )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Above we have a dataframe containing `null` is used to indicate missing data in any data types. For the purposes of this guide we won't mention all pelicularities that may come from alternative dataframes found in other packages such as Pandas.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | Disclamier
        type: info
    [`NaN`](https://docs.pola.rs/user-guide/expressions/missing-data/#not-a-number-or-nan-values) can also be used as a missing data placeholder in Polars for missing values in datasets with floats, especially in formats that lack proper support for null values, but it is not missing value per se (the same way you may see some datasets give 0 or -1 a distinct meaning) see the documentation for more.


    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Detecting Nulls

    Use `is_null` or `null_count` to find missing values. As well as `==` (`eq()`) to return `null` when comparing two null values
    `.eq_missing()` returning true when comparing two null values
    """
    )
    return


@app.cell(hide_code=True)
def _(df, mo, pl):
    # Show where nulls are
    mask = df.select(pl.all().is_null().name.suffix('_is_null'))
    counts = df.null_count()
    # mo.vstack([mask, counts])

    mo.vstack([
        mo.md("### Null Detection Results"),
        mo.md("#### Boolean Mask of Null Values"),
        mask,
        mo.md("#### Count of Null Values per Column"),
        counts
    ])
    return


@app.cell(hide_code=True)
def _(df, mo, pl):


    # Compare using == operator (same as .eq())
    eq_result = df.with_columns([
        (pl.col("fruits") == pl.col("fruits")).alias("A_eq_B")
    ])

    # Compare using .eq_missing()
    eq_missing_result = df.with_columns([
                pl.col("score").eq_missing(pl.col("B")).alias("A_eq_missing_B")
    ])

    mo.vstack([
        mo.md("### Comparison Results"),
        mo.md("#### Original DataFrame"),
        df,
        mo.md("#### Using == operator (eq())"),
        eq_result,
        mo.md("#### Using .eq_missing()"),
        eq_missing_result
    ])
    return


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
def _(df, mo):
    # drop any row with a null
    dropped_all = df.drop_nulls()
    # drop rows with nulls only in 'score'
    dropped_score = df.drop_nulls(subset=["score"])
    mo.vstack([
            mo.md("### Dropping Null Values"),
            mo.md("#### Drop All Rows with Any Null Value"),
            dropped_all,
            mo.md("#### Drop Rows with Null Values Only in 'score' Column"),
            dropped_score
    ])
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | Sidenote
        type: info

        Theres also an alternative function ```interpolate``` that can be used to fill missing values in a column with the average of the surrounding values. This is useful when you have a time series or a sequence of values and you want to fill in the gaps with a smooth transition.

    ///
    """
    )
    return


@app.cell
def _(df, mo):
    filled_constant = df.fill_null(0)
    filled_ffill = df.fill_null(strategy="forward")
    filled_bfill = df.fill_null(strategy="backward")
    mo.vstack([
            mo.md("### Filling Null Values"),
            mo.md("#### Fill with Constant Value (0)"),
            filled_constant,
            mo.md("#### Forward Fill (Using Previous Non-null Value)"),
            filled_ffill,
            mo.md("#### Backward Fill (Using Next Non-null Value)"),
            filled_bfill
        ])
    return


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
def _(df, mo, pl):
    mean_age = df.select(pl.col("age").mean()).item()
    imputed = df.with_columns(
        pl.col("age").fill_null(mean_age).alias("age_imputed")
    )
    mo.vstack([
            mo.md("### Imputing with Expressions"),
            mo.md(f"#### Mean Age: {mean_age}"),
            mo.md("#### DataFrame with Imputed Age Values"),
            imputed
        ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 🔖 References

    - Handling missing data in Polars: https://docs.pola.rs/user-guide/expressions/missing-data/
    - Handling NaN datatypes : https://docs.pola.rs/user-guide/expressions/missing-data/#not-a-number-or-nan-values
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

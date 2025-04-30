# /// script
# dependencies = [
#     "marimo",
#     "polars==1.28.1",
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
    return mo, pl


@app.cell
def _(pl):
    base = pl.DataFrame(
        {"id": [1, 2, 3, 4], "name": ["Alice", "Bob", "Charlie", "Diana"]}
    )

    other = pl.DataFrame({"id": [3, 4, 5, 6], "age": [25, 32, 40, 28]})
    return base, other


@app.cell
def _(base, mo, other):
    mo.vstack(
        [
            mo.hstack([mo.ui.table(base, show_download=False , label='base'), mo.ui.table(other, show_download=False, label='other')], justify="center"),
        ]
    )
    return


@app.cell
def _(mo):
    join_options: dict = {
        "inner (default)": "inner",
        "left": "left",
        "right": "right",
        "full": "full",
        "semi": "semi",
        "anti": "anti",
    }

    # These are from https://docs.pola.rs/user-guide/transformations/joins/#quick-reference-table
    descriptions: dict = {
        "inner": "Keeps rows that matched both on the left and right.",
        "left": "Keeps all rows from the left plus matching rows from the right. Non-matching rows from the left have their right columns filled with null.",
        "right": "Keeps all rows from the right plus matching rows from the left. Non-matching rows from the right have their left columns filled with null.",
        "full": "Keeps all rows from either dataframe, regardless of whether they match or not. Non-matching rows from one side have the columns from the other side filled with null.",
        "semi": "Keeps rows from the left that have a match on the right.",
        "anti": "Keeps rows from the left that do not have a match on the right.",
        "join_where": "Finds all possible pairings of rows from the left and right that satisfy the given predicate(s).",
        "join_asof": "Like a left outer join, but matches on the nearest key instead of on exact key matches.",
        "cross": "Computes the Cartesian product of the two dataframes.",
    }
    dropdown = mo.ui.dropdown(
        value="inner (default)",
        options=join_options,
    )
    return descriptions, dropdown


@app.cell
def _(base, descriptions, dropdown, mo, other, pl):
    result: pl.DataFrame = base.join(other, on="id", how=dropdown.value)
    n_rows, n_columns = result.shape
    mo.vstack(
        [
            dropdown,
            mo.md(descriptions[dropdown.value]),
            mo.md(f"rows: {n_rows} columns: {n_columns}"),
        ]
    )
    return (result,)


@app.cell
def _(mo, result):
    mo.vstack([result])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

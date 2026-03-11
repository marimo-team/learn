# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "marimo-learn>=0.7.0",
#     "polars==1.24.0",
#     "sqlalchemy",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import marimo_learn as mol
    from marimo_learn import MultipleChoiceWidget, OrderingWidget
    import sqlalchemy

    db_path = mol.localize_file("penguins.db")
    DATABASE_URL = f"sqlite:///{db_path}"
    engine = sqlalchemy.create_engine(DATABASE_URL)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Basic Selection

    This tutorial shows how to select values from a single table in a database using SQL. We have already made a connection between this notebook and our `penguins.db` database—we'll show you how to do that later—so let's have a look at the data in the `penguins` table.
    """)
    return


@app.cell
def _():
    _df = mo.sql(
        f"""
        select * from penguins;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Almost every **query** in SQL starts with the word `select`. The value immediately after it tells the database manager what we want to see. In this case, we use the shorthand `*` to mean "all the columns". We then say `from penguins` to tell the database manager which table we want to get the data from. The semi-colon at the end marks the end of the query.

    Note that the database manager doesn't format the output nicely, draw the little distribution histograms above columns, or give us the page-forward/page-backward controls: all the credit for that belongs to the Marimo notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Choosing Columns

    We don't have to select all of the columns every time we get data from a table. If we only want specific columns, we give their names instead of using `*` to mean "all". As the output below shows, the columns are displayed in the order in which we gave their names.
    """)
    return


@app.cell
def _():
    _df = mo.sql(
        f"""
        select sex, island, species from penguins;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Try editing the SQL in the query cell to change the column order, or to get the `bill_length_mm` column.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Upper and Lower Case

    We can write the query above in any mixture of upper and lower case and get the same result.
    """)
    return


@app.cell
def _():
    _df = mo.sql(
        f"""
        SELECT Sex, island, SPECIES frOM pEnGuInS;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please don't do this: it makes your queries very hard to read. It *is* common to use upper case for keywords like `SELECT` and `FROM`, and lower case for column names like `penguins` and `island`; whatever you choose, the most important thing is to be consistent.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sorting

    When we look at a spreadsheet or a printed table, the rows are in a particular order. A database manager, on the other hand, might rearrange rows for the sake of efficiency as data is added or deleted, which means the rows displayed by `select` can be in whatever order it wants. If we want a particular order, we can add `order by` and the names of one or more columns to our query.

    Note that we have split the query below across several lines to make it easier to read. Just as SQL doesn't care about upper and lower case, it doesn't care about line breaks. As our queries become larger and more complicated, formatting them like this will make them a lot easier to understand.
    """)
    return


@app.cell
def _():
    _df = mo.sql(
        f"""
        select island, species, sex
        from penguins
        order by island, species;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you page through the output from the query above, you'll see that our penguins have been ordered by island: Biscoe before Dream, and Dream before Torgersen. Within each of those groups, the penguins are sub-ordered by species (Adelie, Chinstrap, and then Gentoo). The penguins aren't ordered by sex, but they could be: as with island and species, the sorting goes from left to right.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Try rearranging the order of columns in the `select` while leaving the order in `order by` alone and vice versa. Notice that you don't have to sort in the order in which the columns are displayed (but you usually should to make the output easier to understand).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > What do you think will happen if you select `island` and `species` but `order by sex`? How can you tell if your prediction is correct?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Limiting Output

    The `penguins` table has 344 rows. If we only want to see the first five, we can add a `limit` clause to our query, which specifies the maximum number of rows we want.
    """)
    return


@app.cell
def _():
    _df = mo.sql(
        f"""
        select * from penguins limit 5;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    What if we want the next five? Or the five after that? To get those, we can add an offset, which is the number of rows to skip before selecting as many rows as we've asked for.
    """)
    return


@app.cell
def _():
    _df = mo.sql(
        f"""
        select * from penguins
        limit 5 offset 5;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Selecting one chunk of data after another is called **paging**. Applications frequently do this in order to save memory and bandwidth: people can't look at 100,000 rows at once, so there's usually no point grabbing that many in one gulp.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Add a cell below to get rows 12 through 17 from the `penguins` table. Think carefully about what the `offset` and `limit` need to be to get precisely these rows.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Try changing the query above to be `offset 5 limit 5`. Do you understand the result?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > 1. What happens if you specify a limit that is greater than the number of rows in the table?
    > 1. What happens if you specify an offset that is greater than the number of rows in the table?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Suppose your program is paging through a table while another application is adding and deleting rows. What would you want to happen? What do you think will happen?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Removing Duplicates

    Suppose we want to find out which kinds of penguins were seen on which islands. We could scroll through the data, taking note of each unique (species, island) pair we see, but SQL will do this for us if we add the `distinct` keyword to our query.

    Note that the query below includes a comment explaining what it does. While comments in Python start with `#`, comments in SQL start with `--` and run to the end of the line.
    """)
    return


@app.cell
def _():
    _df = mo.sql(
        f"""
        -- Show unique (species, island) pairs.
        select distinct species, island
        from penguins;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Modify the query above to show (island, species) instead of (species, island), and to sort by island name and then by species name.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Doing Calculations

    The `penguins` table records the penguins' masses in grams (at least, that's what we think the `_g` suffix on the column name means). If we want the mass in kilograms, we can divide the given values by 1000.
    """)
    return


@app.cell
def _():
    _df = mo.sql(
        f"""
        select species, sex, body_mass_g, body_mass_g / 1000
        from penguins
        limit 10;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The query above shows both the mass in grams and the mass in kilograms so that we can check the latter against the former. However, the name that the database manager automatically gives the calculated column isn't particular readable. Let's use `as` to fix that.
    """)
    return


@app.cell
def _():
    _df = mo.sql(
        f"""
        select species, sex, body_mass_g, body_mass_g / 1000 as mass_kg
        from penguins
        limit 10;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Can you use `as` to select a column from the table but display it with a different name? Should you?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Write a query to calculate the ratio of bill length and bill height for every penguin. Call the calculated column `bill_ratio`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Check Understanding

    ![concept map](public/01_concepts.svg)
    """)
    return


@app.cell(hide_code=True)
def _():
    _widget = mo.ui.anywidget(
        OrderingWidget(
            question="Arrange these SQL clauses in the order they must appear in a query.",
            items=["SELECT", "FROM", "ORDER BY", "LIMIT"],
        )
    )
    _widget
    return


@app.cell(hide_code=True)
def _():
    _widget = mo.ui.anywidget(
        MultipleChoiceWidget(
            question="What does `SELECT *` mean in a SQL query?",
            options=[
                "Select only the first row of the table",
                "Select all columns from the table",
                "Select all rows but only the first column",
                "Count the total number of rows",
            ],
            correct_answer=1,
            explanation="`*` is shorthand for 'all columns'. `SELECT *` retrieves every column; the number of rows returned depends on whether you add WHERE, LIMIT, or other clauses.",
        )
    )
    _widget
    return


if __name__ == "__main__":
    app.run()

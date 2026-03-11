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


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import marimo_learn as mol
    import sqlalchemy

    db_path = mol.localize_file("lab.db")
    DATABASE_URL = f"sqlite:///{db_path}"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return engine, mo, mol


@app.cell(hide_code=True)
def _():
    from marimo_learn import LabelingWidget, OrderingWidget
    return LabelingWidget, OrderingWidget


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Combining Tables

    Relational databases get their name from the fact that they store the relations between tables. This tutorial shows how to connect and combine information from multiple tables. We will save most of the exercises for the next tutorial, where we start working with our first complex database.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Basic Joins

    The `jobs` database has two tables. The first, called `job`, shows the credits that students can earn doing different kinds of jobs, and has two rows and two columns:

    | name | credits |
    | :--- | ------: |
    | calibrate | 1.5 |
    | clean | 0.5 |

    The other table, `work`, keeps track of who has done which jobs:

    | person | job |
    | :----- | :-- |
    | Amal | calibrate |
    | Amal | clean |
    | Amal | complain |
    | Gita | clean |
    | Gita | clean |
    | Gita | complain |
    | Madhi | complain |

    We want to know how many credits each student has earned. The first step in answering this is to **join** the tables together.
    """)
    return


@app.cell
def _(job, work):
    _df = mo.sql(
        f"""
        select *
        from job join work;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `join` operation creates a temporary table in memory by combining every row of `job` with every row of `work`. Since `job` has two rows and `work` has seven, the temporary table has 2×7=14 rows.

    Some of these rows are useful: the first, for example, tells us that Amal did some calibration, and that calibrating is worth 1.5 credits. The second, however, combines information about calibrating with the fact that Amal did some cleaning. We can get rid of the rows that aren't useful by filtering with `where`.
    """)
    return


@app.cell
def _(job, work):
    _df = mo.sql(
        f"""
        select *
        from job join work
        where job.name = work.job;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This query demonstrates two things:

    1. When we are working with two or more tables, we refer to columns using `table_name.column_name`, as in `job.name` or `work.job`. We don't absolutely need to do this in this query, since columns' names are all unique, but it's very common to have columns with the same names in different tables. In those cases the two-part names are required to avoid ambiguity; it is therefore good practice to *always* use two-part names when working with multiple tables.
    2. There isn't an entry in `job` for `complain`, so `job.name = work.job` isn't true for any of the combined rows that involve complaining. On the other hand, Gita cleaned up the lab twice, so there are two records in the result for that. This shows that `join` doesn't automatically remove duplicates.

    While we can use `where`, the SQL standard encourages us to use a different keyword `on`:
    """)
    return


@app.cell
def _(job, work):
    _df = mo.sql(
        f"""
        select *
        from job join work
        on job.name = work.job;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Many years ago, using `on` sometimes gave slightly higher performance. Today, though, the two forms are equivalent from the database manager's point of view. Many people still prefer `on` for readability: it shows how the rows are being combined, while `where` shows how combined rows are being filtered. As with almost everything in programming, what matters most is to pick one and stick to it so that your queries are consistent.

    The standard also encourages us to write our join as `inner join`, because as we will see in a moment, other kinds of joins exist. People often skip this and just write `join`, or even use a simple comma between the table names, but from now on we will be pedantic to make what we're doing clearer.

    We are now able to answer our original question: how many credits has each student earned?
    """)
    return


@app.cell
def _(job, work):
    _df = mo.sql(
        f"""
        select work.person, sum(job.credits) as total     -- add up the credits for each person
        from job inner join work                          -- notice: inner join
        on job.name = work.job
        group by work.person;                             -- put all the credits for each person into a separate bucket
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Left Joins

    The query above shows us how many credits Amal and Gita have earned, but doesn't show anything for Madhi. Ideally, we'd like a row showing that she has earned zero credits. To get this, we need to use a different kind of join called a **left join**. A left join is created by following these rules:

    1. If the row from the left-hand table matches one or more rows from the right-hand table, combine them as an inner join would.
    2. If the row from the left-hand table _doesn't_ match any rows from the right-hand table, create one row in the result with the values from the left row and `null` where the values from the right-hand table would be.

    An example will make this clearer.
    """)
    return


@app.cell
def _(job, work):
    _df = mo.sql(
        f"""
        select *
        from work left join job
        on work.job = job.name;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's trace this query's execution step by step:

    1. The `(Amal, calibrate)` row from `work` matches the `(calibrate, 1.5)` row from `job`, so that is the first row of output.
    2. Similarly, the `(Amal, clean)` row matches the `(clean, 0.5)` row, so we get the second row of output.
    3. But `(Amal, complain)` _doesn't_ match anything from `job`, so we get a row with the values from the left table (`Amal` and `complain`) and `null` for `name` and `work`.
    4. We then get two rows for Gita cleaning because there's a match…
    5. …and two rows with `null` values for Gita and Madhi complaining because there isn't.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > What do we get if we invert the order of the tables, i.e., do `job left join work`? Why?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Coalesce

    We can now sum up everyone's credits:
    """)
    return


@app.cell
def _(job, work):
    _df = mo.sql(
        f"""
        select work.person, sum(job.credits) as total
        from work left join job                          -- notice: left join
        on work.job = job.name
        group by work.person;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This is *almost* what we want: we have a row for Madhi, but her `total` is `null` because that's what `sum` produces when all of the values it's adding up are `null`. We can fix this using a built-in SQL function called `coalesce`:
    """)
    return


@app.cell
def _(job, work):
    _df = mo.sql(
        f"""
        select
            work.person,
            coalesce(sum(job.credits), 0) as total
        from
            work left join job
        on
            work.job = job.name
        group by
            work.person;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `coalesce` takes two inputs. If the first is not `null`, `coalesce` returns that. If the first value *is* `null`, on the other hand, `coalesce` returns its second input. In simpler terms, it gives us a value or a default if the value is `null`.

    Note that we have split this query across several lines with the keywords at the left margin and the parts of the query that belong to them indented below them. As our queries become more complex, this style makes them easier to read. As with `join` versus `inner join`, the most important thing is to be consistent so that the reader isn't distracted by stylistic differences.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Check Understanding

    ![concept map](/public/05_concepts.svg)
    """)
    return


@app.cell(hide_code=True)
def _(OrderingWidget, mo):
    _widget = mo.ui.anywidget(
        OrderingWidget(
            question="Arrange the steps SQL follows when executing an INNER JOIN.",
            items=[
                "Combine every row from the left table with every row from the right table",
                "Apply the ON condition to keep only matching row pairs",
                "Apply any WHERE clause to filter the matched rows further",
                "Apply SELECT to return only the requested columns",
            ],
        )
    )
    _widget
    return


@app.cell(hide_code=True)
def _(LabelingWidget, mo):
    _widget = mo.ui.anywidget(
        LabelingWidget(
            question="Drag each label to the line of the query it best describes.",
            labels=["left table", "right table", "join condition", "fallback for null"],
            text_lines=[
                "from work left join job",
                "on work.job = job.name",
                "coalesce(sum(job.credits), 0) as total",
            ],
            correct_labels={0: [0, 1], 1: [2], 2: [3]},
        )
    )
    _widget
    return


if __name__ == "__main__":
    app.run()

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

    db_path = mol.localize_file("penguins.db")
    DATABASE_URL = f"sqlite:///{db_path}"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return engine, mo, mol


@app.cell(hide_code=True)
def _():
    from marimo_learn import MatchingWidget, MultipleChoiceWidget
    return MatchingWidget, MultipleChoiceWidget


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Filtering

    The previous tutorial showed how to select specific columns from a database table, and how to page through the data that a query returns. However, people almost always **filter** data based on its properties rather than on its position in a table. To see how this works, let's look at the combinations of species, island, and sex in the `penguins` table.
    """)
    return


@app.cell
def _(penguins):
    _df = mo.sql(
        f"""
        select distinct species, island, sex from penguins;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Equality

    Suppose we only want to see penguins from Dream island, regardless of their species or sex. To get this, we add a `where` clause to our query.
    """)
    return


@app.cell
def _(penguins):
    _df = mo.sql(
        f"""
        select distinct species, island, sex 
        from penguins
        where island = "Dream";
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    There are several noteworthy things in this query:

    1. We don't have to use `distinct`. If we leave it out, we get *all* the penguins on Dream island. (We included it to make the output easier to read without paging.)
    2. The `where` clause *must* come after the `from` clause. SQL is very picky about ordering…
    3. We don't put quotation marks around `island` because it's the name of a column. We *do* put quotes around `"Dream"` because it's an actual literal piece of text.
    4. We use a single equals sign `=` to check for equality. This is different from most programming languages, which use `==`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Write a query to select all the Chinstrap penguins regardless of what island they're on.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > 1. Change the column name `island` to `ISLAND` and re-run the query: what happens?
    > 2. Change the text value `"Dream"` to `"DREAM"`: what happens?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Comparisons

    We can do all of the usual comparisons in SQL:

    | name | symbol | example |
    | :--- | ------ | :------ |
    | less than | `<` | `body_mass_g < 3300` |
    | less than or equal | `<=` | `flipper_length_mm < 200.0` |
    | equal | `=` | `species = "Gentoo"` |
    | not equal | `!=` or `<>` | `species != "Gentoo"` |
    | greater than or equal | `>=` | `flipper_length_mm >= 200.0` |
    | greater than | `>` | `body_mass_g > 3300` |

    Comparing numbers is straightforward. When we compare text, the comparison uses dictionary order: A is less than B, AA is than AB, and so on.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Find all the penguins that _aren't_ on Torgersen island.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Use `where`, `order by`, and `limit` to find the heaviest penguin. Use it again to find the lightest.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > What happens if we accidentally compare a number to text? For example, what happens if we select penguins where `species` is less than 3000, or where `body_mass_g` is greater than the letter 'M'?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Combining Conditions

    We can combine conditions using `and` and `or`. `and` is the simpler of the two: when we write `where condition_1 and condition_2`, we get the rows where *both* conditions are true.
    """)
    return


@app.cell
def _(penguins):
    _df = mo.sql(
        f"""
        select * from penguins
        where species = 'Gentoo' and body_mass_g > 6000.0;
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If we use `or`, we get rows where *either or both* condition is true. This is different from common English usage: if you tell a child that they can have an ice cream cone or a chocolate bar, you mean "either/or". When you use `or` in SQL, on the other hand, it means "if any of the conditions is true". For example, the query below gets all of the penguins on Biscoe island, as well as all of the Gentoo penguins. Some penguins satisfy both conditions (the Adelie penguins on Biscoe island), some satisfy just one (the Adelies on Torgersen and the Gentoos on Biscoe). Penguins that don't satisfy either, like Chinstrap penguins on Dream island, don't show up at all.
    """)
    return


@app.cell
def _(penguins):
    _df = mo.sql(
        f"""
        select distinct species, island from penguins
        where species = 'Adelie' or island = 'Biscoe';
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We have written our `where` conditions as we would say them. Many programmers would wrap each condition in parentheses to make them easier to read.
    """)
    return


@app.cell
def _(penguins):
    _df = mo.sql(
        f"""
        select distinct species, island from penguins
        where (species = 'Adelie') or (island = 'Biscoe');
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The more complex our conditions are, the more important it is to use parentheses to make sure everyone reading the query (including ourselves) understands what it means. The query below shows an example.
    """)
    return


@app.cell
def _(penguins):
    _df = mo.sql(
        f"""
        select distinct species, island from penguins
        where ((species = 'Adelie') and (island = 'Biscoe')) or (species = 'Chinstrap');
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Explain in simple terms what the condition in the query above is selecting.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > We can put `not` in front of a condition to invert its meaning. Use this to write a query that fetches the same rows as one with the condition `species != 'Chinstrap'`, but which uses `=` instead of `!=`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > Does the expression `species not = 'Gentoo'` work?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > 1. Write a query to find all of the penguins whose bill length is greater than their bill depth.
    > 2. Write another query to find all of the penguins whose bill length is less than their bill depth. What do you notice about the output of this query?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > The previous tutorial showed how to do calculations on the fly to (for example) produce a column called `mass_kg` showing the body mass of each penguin in kilograms. Can these on-the-fly columns be used in `where` conditions? To find out, write a query that finds all of the penguins that weight more than 4.0 kg.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Check Understanding

    ![concept map](/public/02_concepts.svg)
    """)
    return


@app.cell(hide_code=True)
def _(MatchingWidget, mo):
    _widget = mo.ui.anywidget(
        MatchingWidget(
            question="Match each SQL comparison operator to its meaning.",
            left=["<", "!=", ">=", "="],
            right=["equal to", "not equal to", "less than", "greater than or equal to"],
            correct_matches={0: 2, 1: 1, 2: 3, 3: 0},
        )
    )
    _widget
    return


@app.cell(hide_code=True)
def _(MultipleChoiceWidget, mo):
    _widget = mo.ui.anywidget(
        MultipleChoiceWidget(
            question="A query uses `WHERE species = 'Adelie' OR island = 'Biscoe'`. Which rows does it return?",
            options=[
                "Only rows where both conditions are true (Adelie penguins on Biscoe)",
                "Rows where either condition is true, or both",
                "Rows where species is Adelie but island is not Biscoe",
                "Rows where island is Biscoe but species is not Adelie",
            ],
            correct_answer=1,
            explanation="In SQL, OR returns every row where at least one condition is true. This includes rows satisfying just the first condition, just the second, or both simultaneously.",
        )
    )
    _widget
    return


if __name__ == "__main__":
    app.run()

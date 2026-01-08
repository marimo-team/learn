# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "polars==1.23.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic operations on data
    _By [Joram Mutenge](https://www.udemy.com/user/joram-mutenge/)._

    In this notebook, you'll learn how to perform arithmetic operations, comparisons, and conditionals on a Polars dataframe. We'll work with a DataFrame that tracks software usage by year, categorized as either Vintage (old) or Modern (new).
    """)
    return


@app.cell
def _():
    import polars as pl

    df = pl.DataFrame(
        {
            "software": [
                "Lotus-123",
                "WordStar",
                "dBase III",
                "VisiCalc",
                "WinZip",
                "MS-DOS",
                "HyperCard",
                "WordPerfect",
                "Excel",
                "Photoshop",
                "Visual Studio",
                "Slack",
                "Zoom",
                "Notion",
                "Figma",
                "Spotify",
                "VSCode",
                "Docker",
            ],
            "users": [
                10000,
                4500,
                2500,
                3000,
                1800,
                17000,
                2200,
                1900,
                500000,
                12000000,
                1500000,
                3000000,
                4000000,
                2000000,
                2500000,
                4500000,
                6000000,
                3500000,
            ],
            "category": ["Vintage"] * 8 + ["Modern"] * 10,
            "year": [
                1985,
                1980,
                1984,
                1979,
                1991,
                1981,
                1987,
                1982,
                1987,
                1990,
                1997,
                2013,
                2011,
                2016,
                2016,
                2008,
                2015,
                2013,
            ],
        }
    )

    df
    return df, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arithmetic
    ### Addition
    Let's add 42 users to each piece of software. This means adding 42 to each value under **users**.
    """)
    return


@app.cell
def _(df, pl):
    df.with_columns(pl.col("users") + 42)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another way to perform the above operation is using the built-in function.
    """)
    return


@app.cell
def _(df, pl):
    df.with_columns(pl.col("users").add(42))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Subtraction
    Let's subtract 42 users to each piece of software.
    """)
    return


@app.cell
def _(df, pl):
    df.with_columns(pl.col("users") - 42)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, you could subtract like this:
    """)
    return


@app.cell
def _(df, pl):
    df.with_columns(pl.col("users").sub(42))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Division
    Suppose the **users** values are inflated, we can reduce them by dividing by 1000. Here's how to do it.
    """)
    return


@app.cell
def _(df, pl):
    df.with_columns(pl.col("users") / 1000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or we could do it with a built-in expression.
    """)
    return


@app.cell
def _(df, pl):
    df.with_columns(pl.col("users").truediv(1000))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we didn't care about the remainder after division (i.e remove numbers after decimal point) we could do it like this.
    """)
    return


@app.cell
def _(df, pl):
    df.with_columns(pl.col("users").floordiv(1000))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multiplication
    Let's pretend the *user* values are deflated and increase them by multiplying by 100.
    """)
    return


@app.cell
def _(df, pl):
    (df.with_columns(pl.col("users") * 100))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Polars also has a built-in function for multiplication.
    """)
    return


@app.cell
def _(df, pl):
    df.with_columns(pl.col("users").mul(100))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So far, we've only modified the values in an existing column. Let's create a column **decade** that will represent the years as decades. Thus 1985 will be 1980 and 2008 will be 2000.
    """)
    return


@app.cell
def _(df, pl):
    (df.with_columns(decade=pl.col("year").floordiv(10).mul(10)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could create a new column another way as follows:
    """)
    return


@app.cell
def _(df, pl):
    df.with_columns((pl.col("year").floordiv(10).mul(10)).alias("decade"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Tip**
    Polars encounrages you to perform your operations as a chain. This enables you to take advantage of the query optimizer. We'll build upon the above code as a chain.

    ## Comparison
    ### Equal
    Let's get all the software categorized as Vintage.
    """)
    return


@app.cell
def _(df, pl):
    (
        df.with_columns(decade=pl.col("year").floordiv(10).mul(10))
        .filter(pl.col("category") == "Vintage")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could also do a double comparison. VisiCal is the only software that's vintage and in the decade 1970s. Let's perform this comparison operation.
    """)
    return


@app.cell
def _(df, pl):
    (
        df.with_columns(decade=pl.col("year").floordiv(10).mul(10))
        .filter(pl.col("category") == "Vintage")
        .filter(pl.col("decade") == 1970)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could also do this comparison in one line, if readability is not a concern

    **Notice** that we must enclose the two expressions between the `&` with parenthesis.
    """)
    return


@app.cell
def _(df, pl):
    (
        df.with_columns(decade=pl.col("year").floordiv(10).mul(10))
        .filter((pl.col("category") == "Vintage") & (pl.col("decade") == 1970))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also use the built-in function for equal to comparisons.
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(pl.col('category').eq('Vintage'))
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Not equal
    We can also compare if something is `not` equal to something. In this case, category is not vintage.
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(pl.col('category') != 'Vintage')
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or with the built-in function.
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(pl.col('category').ne('Vintage'))
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or if you want to be extra clever, you can use the negation symbol `~` used in logic.
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(~pl.col('category').eq('Vintage'))
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Greater than
    Let's get the software where the year is greater than 2008 from the above dataframe.
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(~pl.col('category').eq('Vintage'))
     .filter(pl.col('year') > 2008)
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or if we wanted the year 2008 to be included, we could use great or equal to.
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(~pl.col('category').eq('Vintage'))
     .filter(pl.col('year') >= 2008)
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could do the previous two operations with built-in functions. Here's with greater than.
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(~pl.col('category').eq('Vintage'))
     .filter(pl.col('year').gt(2008))
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And here's with greater or equal to
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(~pl.col('category').eq('Vintage'))
     .filter(pl.col('year').ge(2008))
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note**: For "less than", and "less or equal to" you can use the operators `<` or `<=`. Alternatively, you can use built-in functions `lt` or `le` respectively.

    ### Is between
    Polars also allows us to filter between a range of values. Let's get the modern software were the year is between 2013 and 2016. This is inclusive on both ends (i.e. both years are part of the result).
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(pl.col('category').eq('Modern'))
     .filter(pl.col('year').is_between(2013, 2016))
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Or operator
    If we only want either one of the conditions in the comparison to be met, we could use `|`, which is the `or` operator.

    Let's get software that is either modern or used in the decade 1980s.
    """)
    return


@app.cell
def _(df, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter((pl.col('category') == 'Modern') | (pl.col('decade') == 1980))
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conditionals
    Polars also allows you create new columns based on a condition. Let's create a column *status* that will indicate if the software is "discontinued" or "in use".

    Here's a list of products that are no longer in use.
    """)
    return


@app.cell
def _():
    discontinued_list = ['Lotus-123', 'WordStar', 'dBase III', 'VisiCalc', 'MS-DOS', 'HyperCard']
    return (discontinued_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's how we can get a dataframe of the products that are discontinued.
    """)
    return


@app.cell
def _(df, discontinued_list, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .filter(pl.col('software').is_in(discontinued_list))
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's create the **status** column.
    """)
    return


@app.cell
def _(df, discontinued_list, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .with_columns(pl.when(pl.col('software').is_in(discontinued_list))
                   .then(pl.lit('Discontinued'))
                   .otherwise(pl.lit('In use'))
                   .alias('status')
                   )
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unique counts
    Sometimes you may want to see only the unique values in a column. Let's check the unique decades we have in our DataFrame.
    """)
    return


@app.cell
def _(df, discontinued_list, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .with_columns(pl.when(pl.col('software').is_in(discontinued_list))
                   .then(pl.lit('Discontinued'))
                   .otherwise(pl.lit('In use'))
                   .alias('status')
                   )
     .select('decade').unique()
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, let's find out the number of software used in each decade.
    """)
    return


@app.cell
def _(df, discontinued_list, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .with_columns(pl.when(pl.col('software').is_in(discontinued_list))
                   .then(pl.lit('Discontinued'))
                   .otherwise(pl.lit('In use'))
                   .alias('status')
                   )
     ['decade'].value_counts()
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could also rewrite the above code as follows:
    """)
    return


@app.cell
def _(df, discontinued_list, pl):
    (df
     .with_columns(decade=pl.col('year').floordiv(10).mul(10))
     .with_columns(pl.when(pl.col('software').is_in(discontinued_list))
                   .then(pl.lit('Discontinued'))
                   .otherwise(pl.lit('In use'))
                   .alias('status')
                   )
     .select('decade').to_series().value_counts()
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hopefully, we've picked your interest to try out Polars the next time you analyze your data.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

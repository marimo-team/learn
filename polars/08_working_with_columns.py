# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars==1.18.0",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Working with Columns

        Author: [Deb Debnath](https://github.com/debajyotid2)

        **Note**: The following tutorial has been adapted from the Polars [documentation](https://docs.pola.rs/user-guide/expressions/expression-expansion).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Expressions

        Data transformations are sometimes complicated, or involve massive computations which are time-consuming. You can make a small version of the dataset with the schema you are trying to work your transformation into. But there is a better way to do it in Polars.

        A Polars expression is a lazy representation of a data transformation. "Lazy" means that the transformation is not eagerly (immediately) executed. 

        Expressions are modular and flexible. They can be composed to build more complex expressions. For example, to calculate speed from distance and time, you can have an expression as:
        """
    )
    return


@app.cell
def _(pl):
    speed_expr = pl.col("distance") / (pl.col("time"))
    speed_expr
    return (speed_expr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Expression expansion

        Expression expansion lets you write a single expression that can expand to multiple different expressions. So rather than repeatedly defining separate expressions, you can avoid redundancy while adhering to clean code principles (Do not Repeat Yourself - [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)). Since expressions are reusable, they aid in writing concise code.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""For the examples in this notebook, we will use a sliver of the *AI4I 2020 Predictive Maintenance Dataset*. This dataset comprises of measurements taken from sensors in industrial machinery undergoing preventive maintenance checks - basically being tested for failure conditions.""")
    return


@app.cell
def _(StringIO, pl):
    data_csv = """
    Product ID,Type,Air temperature,Process temperature,Rotational speed,Tool wear,Machine failure,TWF,HDF,PWF,OSF,RNF
    L51172,L,302.3,311.3,1614,129,0,0,1,0,0,0
    M22586,M,300.8,311.9,1761,113,1,0,0,0,1,0
    L51639,L,302.6,310.4,1743,191,0,1,0,0,0,1
    L50250,L,300,309.1,1631,110,0,0,0,1,0,0
    M20109,M,303.4,312.9,1422,63,1,0,0,0,0,0
    """

    data = pl.read_csv(StringIO(data_csv))
    data
    return data, data_csv


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Function `col`

        The function `col` is used to refer to one column of a dataframe. It is one of the fundamental building blocks of expressions in Polars. `col` is also really handy in expression expansion.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explicit expansion by column name

        The simplest form of expression expansion happens when you provide multiple column names to the function `col`.

        Say you wish to convert all temperature values in deg. Kelvin (K) to deg. Fahrenheit (F). One way to do this would be to define individual expressions for each column as follows:
        """
    )
    return


@app.cell
def _(data, pl):
    exprs = [
        ((pl.col("Air temperature") - 273.15) * 1.8 + 32).round(2),
        ((pl.col("Process temperature") - 273.15) * 1.8 + 32).round(2)
    ]

    result = data.with_columns(exprs)
    result
    return exprs, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Expression expansion can reduce this verbosity when you list the column names you want the expression to expand to inside the `col` function. The result is the same as before.""")
    return


@app.cell
def _(data, pl, result):
    result_2 = data.with_columns(
        (
            (pl.col(
                "Air temperature",
                "Process temperature"
            )
            - 273.15) * 1.8 + 32
        ).round(2)
    )
    result_2.equals(result)
    return (result_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In this case, the expression that does the temperature conversion is expanded to a list of two expressions. The expansion of the expression is predictable and intuitive.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Expansion by data type

        Can we do better than explicitly writing the names of every columns we want transformed? Yes.

        If you provide data types instead of column names, the expression is expanded to all columns that match one of the data types provided.

        The example below performs the exact same computation as before:
        """
    )
    return


@app.cell
def _(data, pl, result):
    result_3 = data.with_columns(((pl.col(pl.Float64) - 273.15) * 1.8 + 32).round(2))
    result_3.equals(result)
    return (result_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        However, you should be careful to ensure that the transformation is only applied to the columns you want. For ensuring this it is important to know the schema of the data beforehand. 

        `col` accepts multiple data types in case the columns you need have more than one data type.
        """
    )
    return


@app.cell
def _(data, pl, result):
    result_4 = data.with_columns(
        (
            (pl.col(
                pl.Float32,
                pl.Float64,
            )
             - 273.15) * 1.8 + 32
        ).round(2)
    )
    result.equals(result_4)
    return (result_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Expansion by pattern matching

        `col` also accepts regular expressions for selecting columns by pattern matching. Regular expressions start and end with ^ and $, respectively.
        """
    )
    return


@app.cell
def _(data, pl):
    data.select(pl.col("^.*temperature$"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Regular expressions can be combined with exact column names.""")
    return


@app.cell
def _(data, pl):
    data.select(pl.col("^.*temperature$", "Tool wear"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Note**: You _cannot_ mix strings (exact names, regular expressions) and data types in a `col` function.""")
    return


@app.cell
def _(data, pl):
    try:
        data.select(pl.col("Air temperature", pl.Float64))
    except TypeError as err:
        print("TypeError:", err)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Selecting all columns

        To select all columns, you can use the `all` function.
        """
    )
    return


@app.cell
def _(data, pl):
    result_6 = data.select(pl.all())
    result_6.equals(data)
    return (result_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Excluding columns

        There are scenarios where we might want to exclude specific columns from the ones selected by building expressions, e.g. by the `col` or `all` functions. For this purpose, we use the function `exclude`, which accepts exactly the same types of arguments as `col`:
        """
    )
    return


@app.cell
def _(data, pl):
    data.select(pl.all().exclude("^.*F$"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`exclude` can also be used after the function `col`:""")
    return


@app.cell
def _(data, pl):
    data.select(pl.col(pl.Int64).exclude("^.*F$"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Column renaming

        When applying a transformation with an expression to a column, the data in the column gets overwritten with the transformed data. However, this might not be the intended outcome in all situations - ideally you would want to store transformed data in a new column. Applying multiple transformations to the same column at the same time without renaming leads to errors.
        """
    )
    return


@app.cell
def _(data, pl):
    from polars.exceptions import DuplicateError

    try:
        data.select(
            (pl.col("Air temperature") - 273.15) * 1.8 + 32,  # This would be named "Air temperature"...
            pl.col("Air temperature") - 273.15,  # And so would this.
        )
    except DuplicateError as err:
        print("DuplicateError:", err)
    return (DuplicateError,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Renaming a single column with `alias`

        The function `alias` lets you rename a single column:
        """
    )
    return


@app.cell
def _(data, pl):
    data.select(
            ((pl.col("Air temperature") - 273.15) * 1.8 + 32).round(2).alias("Air temperature [F]"),
            (pl.col("Air temperature") - 273.15).round(2).alias("Air temperature [C]")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Prefixing and suffixing column names

        As `alias` renames a single column at a time, it cannot be used during expression expansion. If it is sufficient add a static prefix or a static suffix to the existing names, you can use the functions `name.prefix` and `name.suffix` with `col`:
        """
    )
    return


@app.cell
def _(data, pl):
    data.select(
        ((pl.col("Air temperature") - 273.15) * 1.8 + 32).round(2).name.prefix("deg F "),
        (pl.col("Process temperature") - 273.15).round(2).name.suffix(" C"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Dynamic name replacement

        If a static prefix/suffix is not enough, use `name.map`. `name.map` requires a function that transforms column names to the desired. The transformation should lead to unique names to avoid `DuplicateError`.
        """
    )
    return


@app.cell
def _(data, pl):
    # There is also `.name.to_lowercase`, so this usage of `.map` is moot.
    data.select(pl.col("^.*F$").name.map(str.lower))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Programmatically generating expressions

        For this example, we will first create four additional columns with the rolling mean temperatures of the two temperature columns. Such transformations are sometimes used to create additional features for machine learning models or data analysis.
        """
    )
    return


@app.cell
def _(data, pl):
    ext_temp_data = data.with_columns(
            pl.col("^.*temperature$").rolling_mean(window_size=2).round(2).name.prefix("Rolling mean ")
    ).select(pl.col("^.*temperature*$"))
    ext_temp_data
    return (ext_temp_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, suppose we want to calculate the difference between the rolling mean and actual temperatures. We cannot use expression expansion here as we want differences between specific columns.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""At first, you may think about using a `for` loop:""")
    return


@app.cell
def _(ext_temp_data, pl):
    _result = ext_temp_data
    for col_name in ["Air", "Process"]:
        _result = _result.with_columns(
            (abs(pl.col(f"Rolling mean {col_name} temperature") - pl.col(f"{col_name} temperature")))
                .round(2).alias(f"Delta {col_name} temperature")
        )
    _result
    return (col_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Using a `for` loop is functional, but not scalable, as each expression needs to be defined in an iteration and executed serially. Instead we can use a generator in Python to programmatically create all expressions at once. In conjunction with the `with_columns` context, we can take advantage of parallel execution of computations and query optimization from Polars.""")
    return


@app.cell
def _(ext_temp_data, pl):
    def delta_expressions(colnames: list[str]) -> pl.Expr:
        for col_name in colnames:
            yield (abs(pl.col(f"Rolling mean {col_name} temperature") - pl.col(f"{col_name} temperature"))
                            .round(2).alias(f"Delta {col_name} temperature"))


    ext_temp_data.with_columns(delta_expressions(["Air", "Process"]))
    return (delta_expressions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## More flexible column selections

        For more flexible column selections, you can use column selectors from `selectors`. Column selectors allow for more expressiveness in the way you specify selections. For example, column selectors can perform the familiar set operations of union, intersection, difference, etc. We can use the union operation with the functions `string` and `ends_with` to select all string columns and the columns whose names end with "`_high`":
        """
    )
    return


@app.cell
def _(data):
    import polars.selectors as cs

    data.select(cs.string() | cs.ends_with("F"))
    return (cs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Likewise, you can pick columns based on the category of the type of data, offering more flexibility than the `col` function. As an example, `cs.numeric` selects numeric data types (including `pl.Float32`, `pl.Float64`, `pl.Int32`, etc.) or `cs.temporal` for all dates, times and similar data types.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Combining selectors with set operations

        Multiple selectors can be combined using set operations and the usual Python operators:


        | Operator |       Operation      |
        |:--------:|:--------------------:|
        | `A | B`   | Union                |
        | `A & B`    | Intersection         |
        | `A - B`    | Difference           |
        | `A ^ B`    | Symmetric difference |
        | `~A`       | Complement           |

        For example, to select all failure indicator variables excluding the failure variables due to wear, we can perform a set difference between the column selectors.
        """
    )
    return


@app.cell
def _(cs, data):
    data.select(cs.contains("F") - cs.contains("W"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Resolving operator ambiguity

        Expression functions can be chained on top of selectors:
        """
    )
    return


@app.cell
def _(cs, data, pl):
    ext_failure_data = data.select(cs.contains("F")).cast(pl.Boolean)
    ext_failure_data
    return (ext_failure_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        However, operators that perform set operations on column selectors operate on both selectors and on expressions. For example, the operator `~` on a selector represents the set operation “complement” and on an expression represents the Boolean operation of negation.

        For instance, if you want to negate the Boolean values in the columns “HDF”, “OSF”, and “RNF”, at first you would think about using the `~` operator with the column selector to choose all failure variables containing "W". Because of the operator ambiguity here, the columns that are not of interest are selected here.
        """
    )
    return


@app.cell
def _(cs, ext_failure_data):
    ext_failure_data.select((~cs.ends_with("WF")).name.prefix("No"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To resolve the operator ambiguity, we use `as_expr`:""")
    return


@app.cell
def _(cs, ext_failure_data):
    ext_failure_data.select((~cs.ends_with("WF").as_expr()).name.prefix("No"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Debugging selectors

        The function `cs.is_selector` helps check whether a complex chain of selectors and operators ultimately results in a selector. For example, to resolve any ambiguity with the selector in the last example, we can do:
        """
    )
    return


@app.cell
def _(cs):
    cs.is_selector(~cs.ends_with("WF").as_expr())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Additionally we can use `expand_selector` to see what columns a selector expands into. Note that for this function we need to provide additional context in the form of the dataframe.""")
    return


@app.cell
def _(cs, ext_failure_data):
    cs.expand_selector(
        ext_failure_data,
        cs.ends_with("WF"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### References

        1. AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. ([link](https://doi.org/10.24432/C5HS5C)).
        2. Polars documentation ([link](https://docs.pola.rs/user-guide/expressions/expression-expansion/#more-flexible-column-selections))
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import csv
    import marimo as mo
    import polars as pl
    from io import StringIO
    return StringIO, csv, mo, pl


if __name__ == "__main__":
    app.run()

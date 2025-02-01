# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.19"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🔢 Numbers

        This tutorial provides a brief overview of working with numbers.

        ## Number Types

        Python has several types of numbers:

        ```python
        integer = 42          # whole numbers (int)
        decimal = 3.14        # floating-point numbers (float)
        complex_num = 2 + 3j  # complex numbers
        ```

        Below is an example number we'll use to explore operations.
        """
    )
    return


@app.cell
def _():
    number = 42
    return (number,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Basic mathematical operations

        Python supports all standard mathematical operations.

        Try changing the value of `number` above and watch how the results change.
        """
    )
    return


@app.cell
def _(number):
    number + 10  # Addition
    return


@app.cell
def _(number):
    number - 5  # Subtraction
    return


@app.cell
def _(number):
    number * 3  # Multiplication
    return


@app.cell
def _(number):
    number / 2  # Division (always returns float)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Python also has special division operators and power operations.""")
    return


@app.cell
def _(number):
    number // 5  # Floor division (rounds down)
    return


@app.cell
def _(number):
    number % 5  # Modulus (remainder)
    return


@app.cell
def _(number):
    number**2  # Exponentiation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Type conversion

        You can convert between different number types. Try changing these values!
        """
    )
    return


@app.cell
def _():
    decimal_number = 3.14
    return (decimal_number,)


@app.cell
def _(decimal_number):
    int(decimal_number)  # Convert to integer (truncates decimal part)
    return


@app.cell
def _(number):
    float(number)  # Convert to "float" or decimal
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Built-in math functions
        Python provides many useful built-in functions for working with numbers:
        """
    )
    return


@app.cell
def _(number):
    abs(-number)  # Absolute value
    return


@app.cell
def _():
    round(3.14159, 2)  # Round to 2 decimal places
    return


@app.cell
def _():
    max(1, 5, 3, 7, 2)  # Find maximum value
    return


@app.cell
def _():
    min(1, 5, 3, 7, 2)  # Find minimum value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Advanced operations

        For more complex mathematical operations, use Python's [math module](https://docs.python.org/3/library/math.html).
        """
    )
    return


@app.cell
def _():
    import math
    return (math,)


@app.cell
def _(math):
    math.sqrt(16)
    return


@app.cell
def _(math):
    math.sin(math.pi/2)
    return


@app.cell
def _(math):
    math.cos(0)
    return


@app.cell
def _(math):
    math.pi, math.e
    return


@app.cell
def _(math):
    math.log10(100)
    return


@app.cell
def _(math):
    math.log(math.e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next steps:

    - Practice different mathematical operations
    - Experiment with type conversions
    - Try out the math module functions

    Keep calculating! 🧮✨
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

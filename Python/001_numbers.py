# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🔢 Numbers in Python

        Let's explore how Python handles numbers and mathematical operations! 

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
        ## Basic Mathematical Operations

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
    number ** 2  # Exponentiation
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Type Conversion

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
    float(number)  # Convert to float
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Built-in Math Functions
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
        ## Advanced Math Operations

        For more complex mathematical operations, Python's `math` module is your friend:

        ```python
        import math

        # Square root
        math.sqrt(16)    # 4.0

        # Trigonometry
        math.sin(math.pi/2)  # 1.0
        math.cos(0)      # 1.0

        # Constants
        math.pi          # 3.141592653589793
        math.e           # 2.718281828459045

        # Logarithms
        math.log10(100)  # 2.0
        math.log(math.e) # 1.0
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Master the Numbers!

    Next Steps:

    - Practice different mathematical operations

    - Experiment with type conversions

    - Try out the math module functions

    Keep calculating! 🧮✨
    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🧩 Function Design in Python

        Dive into the world of Python functions — where code becomes modular and powerful!

        ## Function Basics
        Functions help you:

        - Break down complex problems

        - Create reusable code blocks

        - Improve code readability (good practice)

        ```python
        def function_name(parameters):
            '''Docstring explaining the function'''
            # Function body
            return result
        ```
        """
    )
    return


@app.cell
def _():
    # Example function with parameter
    def greet(name):
        return f"Hello, {name}!"

    name = "Python Learner"
    return greet, name


@app.cell
def _(greet, name):
    greet(name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Default Parameters
        Make your functions more flexible by providing default values.
        """
    )
    return


@app.cell
def _():
    # Function with default parameter
    def create_profile(name, age=18):
        return f"{name} is {age} years old"

    # Example usage
    example_name = "Alex"
    example_profile = create_profile(example_name)
    example_profile
    return create_profile, example_name, example_profile


@app.cell
def _():
    # Show closure over values
    base_multiplier = 2
    def multiplier(x):
        """
        Create a function that multiplies input by a base value.

        This demonstrates how functions can 'close over' 
        values from their surrounding scope.
        """
        return x * base_multiplier

    # Example of using the closure
    sample_numbers = [1, 2, 3, 4, 5]
    multiplied_numbers = [multiplier(num) for num in sample_numbers]
    print(multiplied_numbers)
    return base_multiplier, multiplied_numbers, multiplier, sample_numbers


@app.cell
def _():
    def calculate(a, b):
        """
        Perform multiple calculations on two numbers.

        Args:
            a (int): First number
            b (int): Second number

        Returns:
            dict: Results of various calculations
        """
        return {
            "sum": a + b,
            "product": a * b,
            "difference": a - b,
            "max": max(a, b)
        }

    # Example usage with concrete values
    first_number = 10
    second_number = 5
    result = calculate(first_number, second_number)
    return calculate, first_number, result, second_number


@app.cell(hide_code=True)
def _(mo, result):
    mo.md(f"""
    ## Function Results

    Calculation Results:
    {result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Multiple Return Values
        Python allows returning multiple values easily:

        ```python
        def multiple_returns():
            return value1, value2, value3

        # Unpacking returns
        x, y, z = multiple_returns()
        ```
        """
    )
    return


@app.cell
def _():
    def weather_analysis(temp):
        """
        Analyze weather based on temperature.

        Args:
            temp (float): Temperature in Celsius

        Returns:
            tuple: Weather status, recommendation, warning level
        """
        if temp <= 0:
            return "Freezing", "Wear heavy coat", "High"
        elif 0 < temp <= 15:
            return "Cold", "Layer up", "Medium"
        elif 15 < temp <= 25:
            return "Mild", "Comfortable clothing", "Low"
        else:
            return "Hot", "Stay hydrated", "High"

    # Example temperature analysis
    temperature = 25
    status, recommendation, warning_level = weather_analysis(temperature)
    return (
        recommendation,
        status,
        temperature,
        warning_level,
        weather_analysis,
    )


@app.cell(hide_code=True)
def _(mo, recommendation, status, warning_level):
    mo.md(f"""
    ## Function Results

    Calculation Results:
    {status}, {recommendation}, {warning_level}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Your Function Design Journey!

    Next Steps:

    - Practice creating functions

    - Experiment with default parameters

    - Explore multiple return values

    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


if __name__ == "__main__":
    app.run()

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
def _(default_age):
    default_age
    return


@app.cell
def _(mo):
    # Interactive function (default parameter demo)
    default_age = mo.ui.number(value=18, start=0, stop=120, label="Default Age")
    return (default_age,)


@app.cell
def _(default_age):
    def create_profile(name, age=default_age.value):
        return f"{name} is {age} years old"

    example_name = "Alex"
    return create_profile, example_name


@app.cell
def _(create_profile, example_name):
    create_profile(example_name)
    return


@app.cell
def _(first_param, mo, second_param):
    mo.hstack([first_param, second_param])
    return


@app.cell
def _(mo):
    # Multiple parameters interactive function demo
    first_param = mo.ui.number(value=10, start=0, stop=100, label="First Number")
    second_param = mo.ui.number(value=5, start=0, stop=100, label="Second Number")
    return first_param, second_param


@app.cell
def _(first_param, second_param):
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

    result = calculate(first_param.value, second_param.value)
    return calculate, result


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
def _(temperature):
    temperature
    return


@app.cell
def _(mo):
    # Multiple return values and how they are captured
    temperature = mo.ui.number(value=25, start=-50, stop=50, label="Temperature")
    return (temperature,)


@app.cell
def _(temperature):
    def weather_analysis(temp):
        """
        Analyze weather based on temperature.

        Args:
            temp (float): Temperature in Celsius (superior unit of measurement)

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

    analysis = weather_analysis(temperature.value)
    return analysis, weather_analysis


@app.cell(hide_code=True)
def _(mo, weather_analysis):
    mo.md(f"""
    ## Multiple Return Demonstration

    Current Temperature Analysis:
    {weather_analysis(25)}
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

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
        # 🧩 Functions

        This tutorial is about an important topic: **functions.**

        A function is a reusable block of code, similar in spirit to a mathematical function. Each function has a **name**, and accepts some number of **arguments**. These arguments are used in the function "body" (its block of code), and each function can **return** values.

        **Example.** Below is an example function.
        """
    )
    return


@app.cell
def _():
    def greet(your_name):
        return f"Hello, {your_name}!"
    return (greet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The keyword `def` starts the function definition. The function's **name** is `greet`. It accepts one **argument** called `your_name`. It then creates a string and **returns** it.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        In the next cell, we **call** the function with a value and assign its return value to a variable.

        **Try it!** Try changing the input to the function.
        """
    )
    return


@app.cell
def _(greet):
    greeting = greet(your_name="<your name here>")
    greeting
    return (greeting,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        **Why use functions?** Functions help you:

        - Break down complex problems
        - Create reusable code blocks
        - Improve code readability
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Default parameters
        Make your functions more flexible by providing default values.
        """
    )
    return


@app.cell
def _():
    def create_profile(name, age=18):
        return f"{name} is {age} years old"
    return (create_profile,)


@app.cell
def _(create_profile):
    # Example usage
    example_name = "Alex"
    example_profile = create_profile(example_name)
    example_profile
    return example_name, example_profile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""You can also create functions that reference variables outside the function body. This is called 'closing over' variables""")
    return


@app.cell
def _():
    base_multiplier = 2

    def multiplier(x):
        """
        Create a function that multiplies input by a base value.

        This demonstrates how functions can 'close over' 
        values from their surrounding scope.
        """
        return x * base_multiplier
    return base_multiplier, multiplier


@app.cell
def _(multiplier):
    print([multiplier(num) for num in [1, 2, 3]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Returning multiple values

        Functions can return multiple values: just separate the values to return by
        commas. Check out the next cell for an example.
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
    return (weather_analysis,)


@app.cell
def _():
    temperature = 25
    return (temperature,)


@app.cell
def _(temperature, weather_analysis):
    status, recommendation, warning_level = weather_analysis(temperature)
    status, recommendation, warning_level
    return recommendation, status, warning_level


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

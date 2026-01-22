# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🔄 Conditional logic

    This tutorial teaches you how to how to make **decisions** in your code, using
    Python's conditional statements.

    ## If Statements
    The foundation of decision-making in Python:
    ```python
    if condition:
        # code to run if condition is True
    elif another_condition:
        # code to run if another_condition is True
    else:
        # code to run if no conditions are True
    ```
    Let's explore with some examples:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Try it!** Try changing the value of `42` below, and see how the output changes.
    """)
    return


@app.cell
def _():
    number = 42
    return (number,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare numbers using operators like

    - `>`
    - `>=`
    - `<`
    - `<=`
    - `==`  (note the two equal signs!)
    """)
    return


@app.cell
def _(mo, number):
    if number > 42:
        result = "Greater than 42"
    elif number == 42:
        result = "Equal to 42!"
    else:
        result = "Less than 42"
    mo.md(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive decision making
    **Try it!** Try changing the conditions below and see how the results change:
    """)
    return


@app.cell(hide_code=True)
def _(mo, threshold, value):
    mo.hstack([value, threshold], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    value = mo.ui.number(value=25, start=0, stop=100, label="Enter a number")
    threshold = mo.ui.slider(value=50, start=0, stop=100, label="Set threshold")
    return threshold, value


@app.cell(hide_code=True)
def _(mo, threshold, value):
    if value.value > threshold.value:
        decision = f"{value.value} is greater than {threshold.value}"
    elif value.value == threshold.value:
        decision = f"{value.value} is equal to {threshold.value}"
    else:
        decision = f"{value.value} is less than {threshold.value}"

    mo.hstack(
        [
            mo.md(f"**Decision**: {decision}"),
            mo.md(
                f"**Threshold cleared?**: {'✅' if value.value >= threshold.value else '❌'}"
            ),
        ],
        justify="space-around",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boolean operations
    Python uses boolean operators to combine conditions:

    - `and`: Both conditions must be True

    - `or`: At least one condition must be True

    - `not`: Inverts the condition
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _text = mo.md("""
        - Try different combinations of age and ID status
        - Notice how both conditions must be True to allow voting
        - Experiment with edge cases (exactly 18, no ID, etc.)
    """)
    mo.accordion({"💡 Experiment Tips": _text})
    return


@app.cell(hide_code=True)
def _(age, has_id, mo):
    mo.hstack([age, has_id], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    age = mo.ui.number(value=18, start=0, stop=120, label="Age")
    has_id = mo.ui.switch(value=True, label="Has ID")
    return age, has_id


@app.cell(hide_code=True)
def _(age, has_id, mo):
    can_vote = age.value >= 18 and has_id.value

    explanation = f"""
    ### Voting eligibility check

    Current Status:

    - Age: {age.value} years old

    - Has ID: {"Yes" if has_id.value else "No"}

    - Can Vote: {"Yes ✅" if can_vote else "No ❌"}

    Reason: {
        "Both age and ID requirements met"
        if can_vote
        else "Missing " + ("required age" if age.value < 18 else "valid ID")
    }
    """

    mo.md(explanation)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Try it!** Write Python code that computes whether an individual can vote.
    """)
    return


@app.cell
def _():
    my_age = 18
    return


@app.cell
def _():
    has_an_id = False
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Complex conditions
    Combine multiple conditions for more sophisticated logic:
    ```python
    # Multiple conditions
    if (age >= 18 and has_id) or has_special_permission:
        print("Access granted")

    # Nested conditions
    if age >= 18:
        if has_id:
            print("Full access")
        else:
            print("Limited access")
    ```
    """)
    return


@app.cell(hide_code=True)
def _(humidity, mo, temp, wind):
    mo.hstack([temp, humidity, wind])
    return


@app.cell(hide_code=True)
def _(mo):
    temp = mo.ui.number(value=25, start=-20, stop=50, label="Temperature (°C)")
    humidity = mo.ui.slider(value=60, start=0, stop=100, label="Humidity (%)")
    wind = mo.ui.number(value=10, start=0, stop=100, label="Wind Speed (km/h)")
    return humidity, temp, wind


@app.cell(hide_code=True)
def _(humidity, mo, temp, wind):
    def get_weather_advice():
        conditions = []

        if temp.value > 30:
            conditions.append("🌡️ High temperature")
        elif temp.value < 10:
            conditions.append("❄️ Cold temperature")

        if humidity.value > 80:
            conditions.append("💧 High humidity")
        elif humidity.value < 30:
            conditions.append("🏜️ Low humidity")

        if wind.value > 30:
            conditions.append("💨 Strong winds")

        return conditions


    conditions = get_weather_advice()

    message = f"""
    ### Weather analysis

    Current Conditions:

    - Temperature: {temp.value}°C

    - Humidity: {humidity.value}%

    - Wind Speed: {wind.value} km/h

    Alerts: {", ".join(conditions) if conditions else "No special alerts"}
    """

    mo.md(message)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next steps

    - Practice combining multiple conditions
    - Explore nested if statements
    - Try creating your own complex decision trees

    Keep coding! 🎯✨
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

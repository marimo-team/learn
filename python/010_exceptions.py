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
        # 🛡️ Handling errors

        Sometimes things go wrong in programs. When that happens, Python raises `exceptions` to tell you what went amiss. For example, maybe you divided by 0:
        """
    )
    return


@app.cell
def _():
    1 / 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        That's a lot of red! The outputs above are Python telling you that
        something went wrong — in this case, we tried dividing a number by 0.

        Python provides tools to catch and handle exceptions: the `try/except`
        block. This is demonstrated in the next couple cells.
        """
    )
    return


@app.cell
def _():
    # Try changing the value of divisor below, and see how the output changes.
    divisor = 0
    return (divisor,)


@app.cell
def _(divisor):
    try:
        print(1 / divisor)
    except ZeroDivisionError as e:
        print("Something went wrong!", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Python has many types of Exceptions besides `ZeroDivisionError`. If you
        don't know what kind of exception you're handling, catch the generic
        `Exception` type:

        ```python
        try:
            ...
        except Exception:
            ...
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(error_types):
    error_types
    return


@app.cell(hide_code=True)
def _(mo):
    # Choose error type
    error_types = mo.ui.dropdown(
        value="ZeroDivisionError",
        options=[
            "ZeroDivisionError", 
            "TypeError", 
            "ValueError", 
            "IndexError", 
            "KeyError"
        ],
        label="Learn about ..."
    )
    return (error_types,)


@app.cell(hide_code=True)
def _(error_types, mo):
    # Error explanation
    error_explanations = {
        "ZeroDivisionError": """
        ### 🚫 ZeroDivisionError
        - Occurs when you try to divide by zero
        - Mathematical impossibility
        - Example:
        ```python
        x = 10 / 0  # Triggers ZeroDivisionError
        ```
        """,
        "TypeError": """
        ### 🔀 TypeError
        - Happens when an operation is applied to an inappropriate type
        - Mixing incompatible types
        - Example:
        ```python
        "2" + 3  # Can't add string and integer
        ```
        """,
        "ValueError": """
        ### 📊 ValueError
        - Raised when a function receives an argument of correct type 
          but inappropriate value
        - Example:
        ```python
        int("hello")  # Can't convert non-numeric string to int
        ```
        """,
        "IndexError": """
        ### 📑 IndexError
        - Occurs when trying to access a list index that doesn't exist
        - Going beyond list boundaries
        - Example:
        ```python
        my_list = [1, 2, 3]
        print(my_list[5])  # Only has indices 0, 1, 2
        ```
        """,
        "KeyError": """
        ### 🗝️ KeyError
        - Raised when trying to access a dictionary key that doesn't exist
        - Example:
        ```python
        my_dict = {"a": 1, "b": 2}
        print(my_dict["c"])  # "c" key doesn't exist
        ```
        """
    }

    mo.md(error_explanations.get(error_types.value, "Select an error type"))
    return (error_explanations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Handling multiple exception types
        
        Catch and handle different types of errors specifically:

        ```python
        def complex_function(x, y):
            try:
                # Potential errors: TypeError, ZeroDivisionError
                result = x / y
                return int(result)
            except TypeError:
                return "Type mismatch!"
            except ZeroDivisionError:
                return "No division by zero!"
            except ValueError:
                return "Conversion error!"
            finally:
                # The `finally` block always runs, regardless if there
                # was an error or not
                ...
                
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(finally_input):
    finally_input
    return


@app.cell(hide_code=True)
def _(mo):
    # Finally Block Demonstration
    finally_input = mo.ui.switch(
        label="Throw an error?",
        value=True
    )
    return (finally_input,)


@app.cell
def _(finally_input, mo):
    def simulate_resource_management():
        try:
            # Simulating a resource-intensive operation
            if not finally_input.value:
                return "🟢 Resource processing successful"
            else:
                raise Exception("Simulated failure")
        except Exception as e:
            return f"🔴 Error: {e}"
        finally:
            return "📦 Resource cleanup completed"


    _result = simulate_resource_management()

    mo.md(f"""
    ### Example: the finally clause

    **Scenario**: {"Normal operation" if not finally_input.value else "An exception was raised"}

    **Result**: {_result}

    Notice how the `finally` block always runs, ensuring cleanup!
    """)
    return (simulate_resource_management,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

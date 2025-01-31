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
        # 🛡️ Error Management in Python

        Welcome to the world of Python error handling - where bugs become learning opportunities! 

        ## Why Error Handling Matters
        Imagine your code as a superhero navigating through the treacherous landscape of potential problems. 
        Error handling is like your hero's shield, protecting your program from unexpected challenges.

        ```python
        # Without error handling
        result = 10 / 0  # 💥 Boom! Unhandled ZeroDivisionError

        # With error handling
        try:
            result = 10 / 0
        except ZeroDivisionError:
            result = "Oops! Can't divide by zero 🛑"
        ```
        """
    )
    return


@app.cell
def _(error_types):
    error_types
    return


@app.cell(hide_code=True)
def _(mo):
    # Choose error type
    error_types = mo.ui.dropdown(
        options=[
            "ZeroDivisionError", 
            "TypeError", 
            "ValueError", 
            "IndexError", 
            "KeyError"
        ],
        label="Select an Error Type to Explore"
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


@app.cell
def _(division_input, divisor_input, mo):
    mo.hstack([division_input, divisor_input])
    return


@app.cell(hide_code=True)
def _(mo):
    # Try-Except work help
    division_input = mo.ui.number(
        value=10, 
        label="Number to Divide", 
        start=-100, 
        stop=100
    )
    divisor_input = mo.ui.number(
        value=0, 
        label="Divisor", 
        start=-100, 
        stop=100
    )
    return division_input, divisor_input


@app.cell
def _(division_input, divisor_input, mo):
    # Safe division function with appropriate error handling
    def safe_divide(numerator, denominator):
        try:
            _result = numerator / denominator
            return f"Result: {_result}"
        except ZeroDivisionError:
            return "🚫 Cannot divide by zero!"
        except Exception as e:
            return f"Unexpected error: {e}"

    # Display result with explanation
    _result = safe_divide(division_input.value, divisor_input.value)

    mo.hstack([
        mo.md(f"**Division**: {division_input.value} ÷ {divisor_input.value}"),
        mo.md(f"**Result**: {_result}")
    ])
    return (safe_divide,)


@app.cell
def _(mo):
    # Multiple Exception Handling
    mo.md(
        """
        ## Multiple Exception Handling
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
        ```
        """
    )
    return


@app.cell
def _(error_chain_input):
    error_chain_input
    return


@app.cell
def _(mo):
    # Try it out
    error_chain_input = mo.ui.text(
        label="Try to break the code",
        placeholder="Enter something tricky..."
    )
    return (error_chain_input,)


@app.cell
def _(error_chain_input, mo):
    # Error chain demonstration
    def tricky_function(input_str):
        try:
            # Simulating a error scenario
            number = int(input_str)
            result = 100 / number
            return f"Success! Result: {result}"
        except ValueError:
            return "❌ Could not convert to number"
        except ZeroDivisionError:
            return "❌ Cannot divide by zero"
        except Exception as e:
            return f"🤯 Unexpected error: {type(e).__name__}"

    result = tricky_function(error_chain_input.value)

    mo.hstack([
        mo.md(f"**Input**: {error_chain_input.value}"),
        mo.md(f"**Result**: {result}")
    ])
    return result, tricky_function


@app.cell
def _(finally_input):
    finally_input
    return


@app.cell
def _(mo):
    # Finally Block Demonstration
    finally_input = mo.ui.switch(
        label="Simulate Resource Management",
        value=True
    )
    return (finally_input,)


@app.cell
def _(finally_input, mo):
    def simulate_resource_management():
        try:
            # Simulating a resource-intensive operation
            if finally_input.value:
                return "🟢 Resource processing successful"
            else:
                raise Exception("Simulated failure")
        except Exception as e:
            return f"🔴 Error: {e}"
        finally:
            return "📦 Resource cleanup completed"

    _result = simulate_resource_management()

    mo.md(f"""
    ### Resource Management Simulation

    **Scenario**: {'Normal operation' if finally_input.value else 'Error scenario'}

    **Result**: {_result}

    Notice how the `finally` block always runs, ensuring cleanup!
    """)
    return (simulate_resource_management,)


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Your Error Handling Journey Continues!

    Next Steps:

    - Practice creating custom exceptions
    - Explore context managers
    - Build robust error-handling strategies

    You're becoming a Python error-handling ninja! 🥷🐍
    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


if __name__ == "__main__":
    app.run()

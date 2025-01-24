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
        # 🧩 Modular Programming in Python

        Unlock the power of organized, reusable, and maintainable code!

        ## Why Modular Programming?
        - Break complex problems into smaller, manageable pieces
        - Improve code readability
        - Enhance code reusability
        - Easier debugging and maintenance
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Standard Library Imports
        Python's standard library provides powerful, pre-built modules:

        ```python
        # String manipulation
        import string

        # Operating system interactions
        import os

        # Date and time handling
        import datetime

        # Mathematical operations
        import math
        ```

        For more details, check the [Python Standard Library Documentation](https://docs.python.org/3/library/)
        """
    )
    return


@app.cell
def _():
    # importing and using standard library modules
    import string
    import os
    import datetime
    import math

    # Example of using imported modules
    def demonstrate_standard_library_usage():
        # String module: get all punctuation
        punctuation_example = string.punctuation

        # OS module: get current working directory
        current_dir = os.getcwd()

        # Datetime module: get current date
        today = datetime.date.today()

        # Math module: calculate square root
        sqrt_example = math.sqrt(16)

        return {
            "Punctuation": punctuation_example,
            "Current Directory": current_dir,
            "Today's Date": today,
            "Square Root Example": sqrt_example
        }

    # Run the demonstration
    module_usage_examples = demonstrate_standard_library_usage()
    module_usage_examples
    return (
        datetime,
        demonstrate_standard_library_usage,
        math,
        module_usage_examples,
        os,
        string,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Import Strategies
        Multiple ways to import and use modules:

        ```python
        # Import entire module
        import math

        # Import specific functions
        from math import sqrt, pow

        # Import with alias
        import math as m
        ```
        """
    )
    return


@app.cell
def _():
    def demonstrate_import_strategies():
        """
        Demonstrate different import strategies using the math module
        """
        # Strategy 1: Import entire module
        import math
        entire_module_result = math.sqrt(25)

        # Strategy 2: Import specific functions
        from math import pow, sqrt
        specific_import_result = pow(2, 3)

        # Strategy 3: Import with alias
        import math as m
        alias_result = m.sqrt(16)

        return {
            "Entire Module Import": entire_module_result,
            "Specific Function Import": specific_import_result,
            "Alias Import": alias_result
        }

    # Run the import strategy demonstration
    import_strategy_examples = demonstrate_import_strategies()
    import_strategy_examples
    return demonstrate_import_strategies, import_strategy_examples


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Code Reusability
        Create functions that can be used across different parts of your project
        """
    )
    return


@app.cell
def _():
    def generate_reusable_functions():
        """
        Demonstrate different types of reusable functions
        """

        def process_text(text):
            '''Reusable text processing function'''
            return text.strip().lower()

        def normalize_number(value, min_val=0, max_val=100):
            '''Normalize a number to a specific range'''
            return max(min_val, min(max_val, value))

        def validate_input(value, type_check=str, min_length=1):
            '''Validate input based on type and minimum length'''
            if not isinstance(value, type_check):
                return False
            return len(str(value)) >= min_length

        # usage
        return {
            "Text Processing": {
                "Example 1": process_text("  John Doe  "),
                "Example 2": process_text("  Example@Email.com  ")
            },
            "Number Normalization": {
                "Oversized Input": normalize_number(150),
                "Negative Input": normalize_number(-10, min_val=-20, max_val=50)
            },
            "Input Validation": {
                "Username Validation": validate_input("john"),
                "Age Validation": validate_input(25, type_check=int)
            }
        }

    # Run the reusable functions demonstration
    reusable_function_examples = generate_reusable_functions()
    reusable_function_examples
    return generate_reusable_functions, reusable_function_examples


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Your Modular Programming Journey!

    Next Steps:

    - Explore Python's standard library

    - Practice different import strategies

    - Design reusable functions

    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


if __name__ == "__main__":
    app.run()

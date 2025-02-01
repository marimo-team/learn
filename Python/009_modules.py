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
        # 🧩 Using modules

        A `module` in Python is a Python file that defines functions and variables. Modules can be `imported` into other Python files, letting you reuse their
        functions and variables.

        We have already seen some modules in previous tutorials, including the `math`
        module. Python comes with many other modules built-in.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## The Python standard library

        Python's "standard library" provides many modules, for many kinds of tasks.

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

        See the [Python standard library documentation](https://docs.python.org/3/library/) for a full reference
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Example""")
    return


@app.cell
def _():
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
        ## Import syntax

        You can import entire modules, and access their functions and variables using dot notation (`math.sqrt`). Or you can import specific members:

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
        ## Third-party packages

        In addition to Python's standard library, there are hundreds of thousands of
        modules available for free on the Python Package index.

        These are distributed as Python "packages", and include packages for
        manipulating arrays of numbers, creating web applications, and more. `marimo`
        itself is a third-party package!

        For installing packages on your machine, we recommend using the [`uv` package manager](https://docs.astral.sh/uv/).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

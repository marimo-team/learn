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
        ## Custom Modules
        Create your own Python modules to organize code:

        ```python
        # math_utils.py
        def add(a, b):
            return a + b

        def multiply(a, b):
            return a * b

        # main.py
        import math_utils
        result = math_utils.add(5, 3)
        ```
        """
    )
    return


@app.cell
def _(module_name):
    module_name
    return


@app.cell
def _(mo):
    # Module naming approaches
    module_name = mo.ui.text(
        value="math_utils", 
        label="Module Name"
    )
    return (module_name,)


@app.cell
def _(mo, module_name):
    def generate_module_content(name):
        """Generate a sample module based on the name"""
        return f"""
        # {name}.py
        def add(a, b):
            '''Add two numbers'''
            return a + b

        def multiply(a, b):
            '''Multiply two numbers'''
            return a * b

        def power(a, b):
            '''Raise a to the power of b'''
            return a ** b
        """

    module_content = generate_module_content(module_name.value)

    mo.md(f"""
    ## Module: {module_name.value}.py

    ```python
    {module_content}
    ```
    """)
    return generate_module_content, module_content


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Import Strategies
        Multiple ways to import and use modules:

        ```python
        # Import entire module
        import math_utils

        # Import specific functions
        from math_utils import add, multiply

        # Import with alias
        import math_utils as mu
        ```
        """
    )
    return


@app.cell
def _(import_strategy):
    import_strategy
    return


@app.cell(hide_code=True)
def _(mo):
    # Import strategy selector
    import_strategy = mo.ui.dropdown(
        options=[
            "Import entire module",
            "Import specific functions",
            "Import with alias"
        ],
        label="Choose Import Strategy"
    )
    return (import_strategy,)


@app.cell
def _(import_strategy, mo):
    def demonstrate_import(strategy):
        if strategy == "Import entire module":
            return "import math_utils\nresult = math_utils.add(5, 3)"
        elif strategy == "Import specific functions":
            return "from math_utils import add, multiply\nresult = add(5, 3)"
        else:
            return "import math_utils as mu\nresult = mu.add(5, 3)"

    import_example = demonstrate_import(import_strategy.value)

    mo.md(f"""
    ## Import examples with code

    ```python
    {import_example}
    ```
    """)
    return demonstrate_import, import_example


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
def _(input_type):
    input_type
    return


@app.cell(hide_code=True)
def _(mo):
    # demo of reusability types
    input_type = mo.ui.dropdown(
        options=[
            "String Processing",
            "Number Manipulation",
            "Data Validation"
        ],
        label="Choose Reusability Scenario"
    )
    return (input_type,)


@app.cell
def _(input_type, mo):
    def generate_reusable_function(func_type):
        if func_type == "String Processing":
            return """
        def process_text(text):
            '''Reusable text processing function'''
            return text.strip().lower()

        # Can be used in multiple contexts
        username = process_text("  John Doe  ")
        email = process_text("  Example@Email.com  ")
            """
        elif func_type == "Number Manipulation":
            return """
            def normalize_number(value, min_val=0, max_val=100):
                '''Normalize a number to a specific range'''
                return max(min_val, min(max_val, value))

            # Consistent number handling across the application
            age = normalize_number(150)  # Returns 100
            temperature = normalize_number(-10, min_val=-20, max_val=50)
            """
        else:
            return """
            def validate_input(value, type_check=str, min_length=1):
                '''Validate input based on type and minimum length'''
                if not isinstance(value, type_check):
                    return False
                return len(str(value)) >= min_length

            # Reusable validation across different input types
            valid_username = validate_input("john")
            valid_age = validate_input(25, type_check=int)
            """

    reusable_code = generate_reusable_function(input_type.value)

    mo.md(f"""
    ## Reusability Example: {input_type.value}

    ```python
    {reusable_code}
    ```
    """)
    return generate_reusable_function, reusable_code


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Your Modular Programming Journey!

    Next Steps:

    - Create your own custom modules

    - Experiment with different import strategies

    - Design reusable functions

    """)
    mo.callout(callout_text, kind="success")
    return (callout_text,)


if __name__ == "__main__":
    app.run()

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
        # 🔄 Loops in Python

        Let's explore how Python helps us repeat tasks efficiently with loops! 

        ## Types of Loops
        Python has two main types of loops:

        ```python
        # For loop - when you know how many times to repeat
        for i in range(5):
            print(i)

        # While loop - when you don't know how many repetitions
        while condition:
            do_something()
        ```

        Let's start with a simple list to explore loops.
        """
    )
    return


@app.cell
def _():
    sample_fruits = ["apple", "banana", "orange", "grape"]
    return (sample_fruits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## For Loop Basics

        The for loop is perfect for iterating over sequences.
        Try changing the `sample_fruits` list above and see how the output changes.
        """
    )
    return


@app.cell
def _(sample_fruits):
    def _print_fruits():
        for _fruit in sample_fruits:
            print(f"I like {_fruit}s!")
    _print_fruits()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
        ## Using enumerate()

        When you need both the item and its position, use enumerate():
        """)
    return


@app.cell
def _(sample_fruits):
    def _print_enumerated():
        for _idx, _fruit in enumerate(sample_fruits):
            print(f"{_idx + 1}. {_fruit}")
    _print_enumerated()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
        ## Range in Loops

        range() is a powerful function for generating sequences of numbers:
        """)
    return


@app.cell
def _():
    def _demonstrate_range():
        print("range(5):", list(range(5)))
        print("range(2, 5):", list(range(2, 5)))
        print("range(0, 10, 2):", list(range(0, 10, 2)))
    _demonstrate_range()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## While Loop Basics

    While loops continue as long as a condition is `True`.""")
    return


@app.cell
def _():
    def _count_up():
        _count = 0
        while _count < 5:
            print(f"Count is {_count}")
            _count += 1
    _count_up()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
        ## Loop Control Statements

        Python provides several ways to control loop execution:

        - `break`: Exit the loop immediately

        - `continue`: Skip to the next iteration

        - `else`: Execute when loop completes normally
        """)
    return


@app.cell
def _():
    def _demonstrate_break():
        for _i in range(1, 6):
            if _i == 4:
                break
            print(_i)
        print("Loop ended early!")
    _demonstrate_break()
    return


@app.cell
def _():
    def _demonstrate_continue():
        for _i in range(1, 6):
            if _i == 3:
                continue
            print(_i)
    _demonstrate_continue()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
        ## Practical Loop Patterns

        Here are some common patterns you'll use with loops:

        ```python
        # Pattern 1: Accumulator
        sum = 0
        for num in [1, 2, 3, 4, 5]:
            sum += num

        # Pattern 2: Search
        found = False
        for item in items:
            if condition:
                found = True
                break

        # Pattern 3: Filter
        filtered = []
        for item in items:
            if condition:
                filtered.append(item)
        ```
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Loop Like a Pro!

    Next Steps:

    - Practice using different types of loops
    - Experiment with loop control statements
    - Try combining loops with lists and conditions

    Keep iterating! 🔄✨
    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


if __name__ == "__main__":
    app.run()

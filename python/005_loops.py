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
        # 🔄 Loops

        Let's learn how Python helps us repeat tasks efficiently with loops.

        A "loop" is a way to execute a block of code multiple times. Python has two 
        main types of loops:

        ```python
        # For loop: when you know how many times to repeat
        for i in range(5):
            print(i)

        # While loop: when you don't know how many repetitions
        while condition:
            do_something()
        ```

        Let's start with a simple list to explore loops. Feel free to modify this list and see how the subsequent outputs change.
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
        ## The for loop

        The for loop is perfect for iterating over sequences.
        Try changing the `sample_fruits` list above and see how the output changes.
        """
    )
    return


@app.cell
def _(sample_fruits):
    for _fruit in sample_fruits:
        print(f"I like {_fruit}s!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Getting the position of an item

        When you need both the item and its position, use `enumerate()`:
        """
    )
    return


@app.cell
def _(sample_fruits):
    for _idx, _fruit in enumerate(sample_fruits):
        print(f"{_idx + 1}. {_fruit}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Iterating over a range of numbers

        `range()` is a powerful function for generating sequences of numbers:
        """
    )
    return


@app.cell
def _():
    print("range(5):", list(range(5)))
    print("range(2, 5):", list(range(2, 5)))
    print("range(0, 10, 2):", list(range(0, 10, 2)))
    return


@app.cell
def _():
    for _i in range(5):
        print(_i)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## The `while` loop

        While loops continue as long as a condition is `True`.
        """
    )
    return


@app.cell
def _():
    _count = 0
    while _count < 5:
        print(f"The count is {_count}")
        _count += 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Controlling loop execution

        Python provides several ways to control loop execution:

        - `break`: exit the loop immediately

        - `continue`: skip to the next iteration

        These can be used with both `for` and `while` loops.
        """
    )
    return


@app.cell
def _():
    for _i in range(1, 6):
        if _i == 4:
            print("Breaking out of the loop.")
            break
        print(_i)
    return


@app.cell
def _():
    for _i in range(1, 6):
        if _i == 3:
            continue
        print(_i)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Practical loop patterns

        Here are some common patterns you'll use with loops:

        ```python
        # Pattern 1: Accumulator
        value = 0
        for num in [1, 2, 3, 4, 5]:
            value += num

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
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next steps

        Check out the official [Python docs on loops and control flow](https://docs.python.org/3/tutorial/controlflow.html).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

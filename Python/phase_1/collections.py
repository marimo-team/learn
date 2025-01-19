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
        # 📦 Collections in Python

        Explore Python's built-in collection types — the essential data structures! 

        ## Lists
        Lists are ordered, mutable sequences. Create them using square brackets:

        ```python
        fruits = ["apple", "banana", "orange"]
        numbers = [1, 2, 3, 4, 5]
        mixed = [1, "hello", 3.14, True]  # Can contain different types
        ```

        Below is an example list we'll use to explore operations.
        """
    )
    return


@app.cell
def _():
    sample_list = [1, 2, 3, 4, 5]
    return (sample_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## List Operations

        Here are common operations you can perform on lists.

        Try changing the values in `sample_list` above and watch the results change.
        """
    )
    return


@app.cell
def _(sample_list):
    len(sample_list)  # List length
    return


@app.cell
def _(sample_list):
    # sample_list.append(6)  # Add item to end
    extended_list = sample_list + [6] # concatenate two lists
    extended_list
    return (extended_list,)


@app.cell
def _(extended_list):
    extended_list[0]  # Access first element
    return


@app.cell
def _(extended_list):
    extended_list[-1]  # Access last element
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Tuples

        Tuples are immutable sequences. They're like lists that can't be changed after creation:
        """
    )
    return


@app.cell
def _():
    coordinates = (10, 20)
    return (coordinates,)


@app.cell
def _(coordinates):
    x, y = coordinates  # Tuple unpacking
    x
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""#### Tuple concatenation""")
    return


@app.cell
def _():
    tuple1 = (1, 2, 3)
    tuple2 = (4, 5, 6)

    tuple3 = tuple1 + tuple2

    tuple3
    return tuple1, tuple2, tuple3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Dictionaries

        Dictionaries store key-value pairs. They're perfect for mapping relationships:
        """
    )
    return


@app.cell
def _():
    person = {
        "name": "John Doe",
        "age": 25,
        "city": "New York"
    }
    return (person,)


@app.cell
def _(person):
    person["name"]  # Access value by key
    return


@app.cell
def _(person):
    person.keys()  # Get all keys
    return


@app.cell
def _(person):
    person.values()  # Get all values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Sets

        Sets are unordered collections of unique elements:
        """
    )
    return


@app.cell
def _():
    numbers_set = {1, 2, 3, 3, 2, 1}  # Duplicates are removed
    return (numbers_set,)


@app.cell
def _(numbers_set):
    numbers_set.add(4)  # Add new element
    numbers_set
    return


@app.cell
def _():
    set1 = {1, 2, 3}
    set2 = {3, 4, 5}
    set1.intersection(set2)  # Find common elements
    return set1, set2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Collection Methods and Operations

        Here are some common operations across collections:

        ```python
        # Lists
        my_list = [1, 2, 3]
        my_list.insert(0, 0)     # Insert at position
        my_list.remove(2)        # Remove first occurrence
        my_list.sort()           # Sort in place
        sorted_list = sorted(my_list)  # Return new sorted list

        # Dictionaries
        my_dict = {"a": 1}
        my_dict.update({"b": 2})  # Add new key-value pairs
        my_dict.get("c", "Not found")  # Safe access with default

        # Sets
        set_a = {1, 2, 3}
        set_b = {3, 4, 5}
        set_a.union(set_b)       # Combine sets
        set_a.difference(set_b)  # Elements in A but not in B
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Collection Mastery Awaits!

    Next Steps:

    - Practice list and dictionary comprehensions
    - Experiment with nested collections
    - Try combining different collection types

    Keep organizing data! 🗃️✨
    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


if __name__ == "__main__":
    app.run()

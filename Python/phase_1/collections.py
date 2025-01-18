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
    sample_list.append(6)  # Add item to end
    sample_list
    return


@app.cell
def _(sample_list):
    sample_list[0]  # Access first element
    return


@app.cell
def _(sample_list):
    sample_list[-1]  # Access last element
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Try marimo's Array Interface!

        Explore how marimo handles arrays with this interactive element:
        """
    )
    return


@app.cell
def _(mo):
    item = mo.ui.text(placeholder="Enter list item")
    items = mo.ui.array([item] * 3, label="Create a list of 3 items")
    return item, items


@app.cell
def _(items, mo):
    mo.hstack(
        [
            items,
            mo.md(f"Your list: {items.value}")
        ],
        justify="space-between"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _text = mo.md("""
        - Try entering different types of items
        - Watch how the list updates in real-time

        This is a great way to experiment with list creation!
    """)
    mo.accordion({"💡 Interactive List Builder Tips": _text})
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
        ### Try marimo's Dictionary Interface!

        Create your own dictionary using marimo's interactive elements:
        """
    )
    return


@app.cell
def _(mo):
    key1 = mo.ui.text(placeholder="Key 1")
    value1 = mo.ui.text(placeholder="Value 1")
    key2 = mo.ui.text(placeholder="Key 2")
    value2 = mo.ui.text(placeholder="Value 2")

    dictionary = mo.ui.dictionary(
        {
            "First key": key1,
            "First value": value1,
            "Second key": key2,
            "Second value": value2,
        }
    )

    return dictionary, key1, key2, value1, value2


@app.cell
def _(dictionary, mo):
    mo.hstack(
        [
            dictionary,
            mo.md(f"Your dictionary: {dictionary.value}")
        ],
        justify="space-between"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _text = mo.md("""Enter key-value pairs to build your dictionary
        
        - Watch how the dictionary updates as you type
        
        - Try different types of values

        This interactive builder helps understand dictionary structure!
    """)
    mo.accordion({"💡 Dictionary Builder Tips": _text})
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

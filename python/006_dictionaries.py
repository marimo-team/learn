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
        # 📚 Dictionaries

        Dictionaries are collections of key-value pairs, with each key associated with a value. The keys are unique, meaning they show up only once.

        ## Creating dictionaries
        Here are a few ways to create dictionaries:

        ```python
        simple_dict = {"name": "Alice", "age": 25}
        empty_dict = dict()
        from_pairs = dict([("a", 1), ("b", 2)])
        ```

        Below is a sample dictionary we'll use to explore operations.
        """
    )
    return


@app.cell
def _():
    sample_dict = {
        "name": "Python",
        "type": "programming language",
        "year": 1991,
        "creator": "Guido van Rossum",
        "is_awesome": True,
    }
    return (sample_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Operations

        Let's explore how to work with dictionaries.

        **Try it!** Try modifying the `sample_dict` above and watch how the results change!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Accessing values by key

        Access values by key using square brackets, like below
        """
    )
    return


@app.cell
def _(sample_dict):
    sample_dict['name'], sample_dict['year']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If you're not sure if a dictionary has a given key, use `get()`:""")
    return


@app.cell
def _(sample_dict):
    sample_dict.get("version", "Not specified"), sample_dict.get("type", "Unknown")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Enumerating dictionary contents

        Python dictionaries come with helpful methods to enumerate keys, values, and pairs.
        """
    )
    return


@app.cell
def _(sample_dict):
    print(list(sample_dict.keys()))
    return


@app.cell
def _(sample_dict):
    print(list(sample_dict.values()))
    return


@app.cell
def _(sample_dict):
    print(list(sample_dict.items()))
    return


@app.cell
def _():
    def demonstrate_modification():
        _dict = {"a": 1, "b": 2}
        print("Original:", _dict)

        # Adding/updating
        _dict.update({"c": 3, "b": 22})
        print("After update:", _dict)

        # Removing
        _removed = _dict.pop("b")
        print(f"Removed {_removed}, Now:", _dict)


    demonstrate_modification()
    return (demonstrate_modification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Dictionary comprehension

        Create dictionaries efficiently with dictionary comprehensions:
        """
    )
    return


@app.cell
def _():
    print({x: x**2 for x in range(5)})
    return


@app.cell
def _():
    print({x: x**2 for x in range(5) if x % 2 == 0})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Nested dictionaries

        Dictionaries can contain other dictionaries, creating complex data structures:
        """
    )
    return


@app.cell
def _():
    nested_data = {
        "users": {
            "alice": {
                "age": 25,
                "email": "alice@example.com",
                "interests": ["python", "data science"],
            },
            "bob": {
                "age": 30,
                "email": "bob@example.com",
                "interests": ["web dev", "gaming"],
            },
        }
    }
    return (nested_data,)


@app.cell
def _(mo, nested_data):
    mo.md(f"Alice's age: {nested_data["users"]["alice"]["age"]}")
    return


@app.cell
def _(mo, nested_data):
    mo.md(f"Bob's interests: {nested_data["users"]["bob"]["interests"]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Common dictionary patterns

        Here are some useful patterns when working with dictionaries:

        ```python
        # Pattern 1: Counting items
        counter = {}
        for item in items:
            counter[item] = counter.get(item, 0) + 1

        # Pattern 2: Grouping data
        groups = {}
        for item in _items:
            key = get_group_key(item)
            groups.setdefault(key, []).append(item)

        # Pattern 3: Caching/Memoization
        cache = {}
        def expensive_function(arg):
            if arg not in cache:
                cache[arg] = compute_result(arg)
            return cache[arg]
        ```
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

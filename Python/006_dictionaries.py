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
        # 📚 Python Dictionaries

        Welcome to the world of Python dictionaries — where data gets organized with keys! 

        ## Creating Dictionaries
        Dictionaries are collections of key-value pairs. Here's how to create them:

        ```python
        simple_dict = {"name": "Alice", "age": 25}  # direct creation
        empty_dict = dict()                         # empty dictionary
        from_pairs = dict([("a", 1), ("b", 2)])    # from key-value pairs
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
        ## Basic Dictionary Operations

        Let's explore how to work with dictionaries.
        Try modifying the `sample_dict` above and watch how the results change!
        """
    )
    return


@app.cell
def _(sample_dict):
    # Accessing values of the dictionary
    def access_dict():
        print(f"Name: {sample_dict['name']}")
        print(f"Year: {sample_dict['year']}")


    access_dict()
    return (access_dict,)


@app.cell
def _(sample_dict):
    # Safe access with get()
    def safe_access():
        print(f"Version: {sample_dict.get('version', 'Not specified')}")
        print(f"Type: {sample_dict.get('type', 'Unknown')}")


    safe_access()
    return (safe_access,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Dictionary Methods

        Python dictionaries come with powerful built-in methods:
        """
    )
    return


@app.cell
def _(sample_dict):
    # Viewing dictionary components
    def view_components():
        print("Keys:", list(sample_dict.keys()))
        print("Values:", list(sample_dict.values()))
        print("Items:", list(sample_dict.items()))


    view_components()
    return (view_components,)


@app.cell
def _():
    # Modifying dictionaries
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
        ## Dictionary Comprehension

        Create dictionaries efficiently with dictionary comprehensions:
        """
    )
    return


@app.cell
def _():
    # Dictionary comprehension examples
    def demonstrate_comprehension():
        # Squares dictionary
        _squares = {x: x**2 for x in range(5)}
        print("Squares:", _squares)

        # Filtered dictionary
        _even_squares = {x: x**2 for x in range(5) if x % 2 == 0}
        print("Even squares:", _even_squares)


    demonstrate_comprehension()
    return (demonstrate_comprehension,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Nested Dictionaries

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
def _(nested_data):
    # Accessing nested data
    def access_nested():
        print("Alice's age:", nested_data["users"]["alice"]["age"])
        print("Bob's interests:", nested_data["users"]["bob"]["interests"])

        # Safe nested access
        def _get_nested(data, *keys, default=None):
            _current = data
            for _key in keys:
                if isinstance(_current, dict):
                    _current = _current.get(_key, default)
                else:
                    return default
            return _current

        print("\nSafe access example:")
        print(
            "Charlie's age:",
            _get_nested(
                nested_data, "users", "charlie", "age", default="Not found"
            ),
        )


    access_nested()
    return (access_nested,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Common Dictionary Patterns

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


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Master the Dictionary!

    Next Steps:

    - Practice different dictionary methods
    - Try creating nested data structures
    - Experiment with dictionary comprehensions
    - Build something using common patterns

    Keep organizing your data! 🗂️✨
    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🔄 Advanced collections

    This tutorials hows advanced patterns for working with collections.

    ## Lists of dictionaries

    A common pattern in data handling is working with lists of dictionaries:
    this is helpful for representing structured data like records or entries.
    """)
    return


@app.cell
def _():
    # Sample data: List of user records
    users_data = [
        {"id": 1, "name": "Alice", "skills": ["Python", "SQL"]},
        {"id": 2, "name": "Bob", "skills": ["JavaScript", "HTML"]},
        {"id": 3, "name": "Charlie", "skills": ["Python", "Java"]}
    ]
    return (users_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Let's explore common operations on structured data.

    **Try it!** Try modifying the `users_data` above and see how the results
    change!
    """)
    return


@app.cell
def _(users_data):
    # Finding users with specific skills
    python_users = [
        user["name"] for user in users_data if "Python" in user["skills"]
    ]
    print("Python developers:", python_users)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Nested data structures

    Python collections can be nested in various ways to represent complex data:
    """)
    return


@app.cell
def _():
    # Complex nested structure
    project_data = {
        "web_app": {
            "frontend": ["HTML", "CSS", "React"],
            "backend": {
                "languages": ["Python", "Node.js"],
                "databases": ["MongoDB", "PostgreSQL"]
            }
        },
        "mobile_app": {
            "platforms": ["iOS", "Android"],
            "technologies": {
                "iOS": ["Swift", "SwiftUI"],
                "Android": ["Kotlin", "Jetpack Compose"]
            }
        }
    }
    return (project_data,)


@app.cell
def _(project_data):
    # Nested data accessing
    backend_langs = project_data["web_app"]["backend"]["languages"]
    print("Backend languages:", backend_langs)

    ios_tech = project_data["mobile_app"]["technologies"]["iOS"]
    print("iOS technologies:", ios_tech)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Example: data transformation

    Let's explore how to transform and reshape collection data:
    """)
    return


@app.cell
def _():
    # Data-sample for transformation
    sales_data = [
        {"date": "2024-01", "product": "A", "units": 100},
        {"date": "2024-01", "product": "B", "units": 150},
        {"date": "2024-02", "product": "A", "units": 120},
        {"date": "2024-02", "product": "B", "units": 130}
    ]
    return (sales_data,)


@app.cell
def _(sales_data):
    # Transform to product-based structure
    product_sales = {}
    for sale in sales_data:
        if sale["product"] not in product_sales:
            product_sales[sale["product"]] = []
        product_sales[sale["product"]].append({
            "date": sale["date"],
            "units": sale["units"]
        })

    print("Sales by product:", product_sales)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## More collection utilities

    Python's `collections` module provides specialized container datatypes:

    ```python
    from collections import defaultdict, Counter, deque

    # defaultdict - dictionary with default factory
    word_count = defaultdict(int)
    for word in words:
        word_count[word] += 1

    # Counter - count hashable objects
    colors = Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])
    print(colors.most_common(2))  # Top 2 most common colors

    # deque - double-ended queue
    history = deque(maxlen=10)  # Only keeps last 10 items
    history.append(item)
    ```
    """)
    return


@app.cell
def _():
    from collections import Counter

    # Example using Counter
    programming_languages = [
        "Python", "JavaScript", "Python", "Java", 
        "Python", "JavaScript", "C++", "Java"
    ]

    language_count = Counter(programming_languages)
    print("Language frequency:", dict(language_count))
    print("Most common language:", language_count.most_common(1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next steps

    For a reference on the `collections` module, see [the official Python
    docs](https://docs.python.org/3/library/collections.html).
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

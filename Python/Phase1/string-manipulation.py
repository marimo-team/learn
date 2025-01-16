# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.12"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🐍 Python Strings: Your First Data Type

        Welcome to your first Python lesson! Today, we'll explore strings - one of Python's fundamental data types.

        ## What are Strings?
        Strings are sequences of characters - like words, sentences, or any text. In Python, we create strings by 
        enclosing text in either single (`'`) or double (`"`) quotes.

        ```python
        greeting = "Hello, Python!"
        name = 'Alice'
        ```
        """
    )
    return


@app.cell
def _(input_text):
    input_text
    return


@app.cell
def _(mo):
    input_text = mo.ui.text(
        value="Hello, World!",
        placeholder="Type any text here...",
        label="Create your first string"
    )
    return (input_text,)


@app.cell
def _(input_text, mo):
    mo.md(f"""
    ### Your String Analysis

    Let's analyze the string you created:

    - Your string: `"{input_text.value}"`

    - Length: `{len(input_text.value)}`

    - First character: `'{input_text.value[0] if input_text.value else ''}'`

    - Last character: `'{input_text.value[-1] if input_text.value else ''}'`
    """)
    return


@app.cell
def _(operation):
    operation
    return


@app.cell
def _(input_text, mo, operation, result):
    mo.md(f"""
    ### String Operation Result

    Original: `{input_text.value}`

    Result: `{result}`

    Python code representation:
    ```python
    text = "{input_text.value}"
    result = text.{operation.selected_key}()
    print(result)  # {result}
    ```
    """)
    return


@app.cell
def _(mo):

    operation = mo.ui.dropdown(
        options={
            "upper": "Convert to UPPERCASE",
            "lower": "Convert to lowercase",
            "title": "Convert To Title Case",
            "strip": "Remove extra spaces"
        },
        value="upper",
        label="Choose a string operation"
    )
    return (operation,)


@app.cell
def _(input_text, operation):
    operations = {
        "Convert to UPPERCASE": input_text.value.upper(),
        "Convert to lowercase": input_text.value.lower(),
        "Convert To Title Case": input_text.value.title(),
        "Remove extra spaces": input_text.value.strip()
    }

    result = operations[operation.value]
    return operations, result


@app.cell(hide_code=True)
def _(mo):
    slice_text = mo.ui.text(
        value="Python",
        placeholder="Enter text to slice",
        label="Text for slicing"
    )

    start_idx = mo.ui.number(
        value=0,
        start=0,
        stop=10,
        label="Start Index"
    )

    end_idx = mo.ui.number(
        value=3,
        start=0,
        stop=10,
        label="End Index"
    )
    return end_idx, slice_text, start_idx


@app.cell(column=1)
def _(end_idx, slice_text, start_idx):
    slice_text, start_idx, end_idx
    return


@app.cell
def _(end_idx, mo, slice_text, start_idx):
    sliced = slice_text.value[start_idx.value:end_idx.value]
    mo.md(f"""
    ### String Slicing

    Text: `{slice_text.value}`

    Slice `[{start_idx.value}:{end_idx.value}]`: `{sliced}`

    ```python
    text = "{slice_text.value}"
    slice = text[{start_idx.value}:{end_idx.value}]
    print(slice)  # {sliced}
    ```
    """)
    return (sliced,)


if __name__ == "__main__":
    app.run()

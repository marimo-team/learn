# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🎭 Strings

        This notebook introduces **strings**, which are containers for text.

        ## Creating strings
        Create strings by wrapping text in quotes:

        ```python
        # Use double quotes
        greeting = "Hello, Python!"

        # or single quotes
        name = 'Alice'

        # or triple quotes
        multiline_string = \"""
        Dear, Alice,
        Nice to meet you.
        Sincerely,
        Bob.
        \"""
        ```

        Below is an example string.
        """
    )
    return


@app.cell
def _():
    text = "Python is amazing!"
    text
    return (text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Essential string operations

        Here are some methods for working with strings.

        Tip: Try changing the value of `text` above, and watch how the
        computed values below change.
        """
    )
    return


@app.cell
def _(text):
    # the `len` method returns the number of characters in the string.
    len(text)
    return


@app.cell
def _(text):
    text.upper()
    return


@app.cell
def _(text):
    text.lower()
    return


@app.cell
def _(text):
    text.title()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Use string methods and the `in` operator to find things in strings.""")
    return


@app.cell
def _(text):
    # Returns the index of "is" in the string
    text.find("is")
    return


@app.cell
def _(text):
    "Python" in text
    return


@app.cell
def _(text):
    "Javascript" in text
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Inserting values in strings

        Modern Python uses f-strings to insert values into strings. For example,
        check out how the next cell greets you (and notice the `f''''`)!

        **Try it!** Enter your name in `my_name` below, then run the cell.
        """
    )
    return


@app.cell
def _():
    my_name = ""
    return (my_name,)


@app.cell
def _(my_name):
    f"Hello, {my_name}!"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Working with parts of strings
        You can access any part of a string using its position (index):
        """
    )
    return


@app.cell
def _(text):
    first_letter = text[0]
    first_letter
    return (first_letter,)


@app.cell
def _(text):
    last_letter = text[-1]
    last_letter
    return (last_letter,)


@app.cell
def _(text):
    first_three = text[0:3]
    first_three
    return (first_three,)


@app.cell
def _(text):
    last_two = text[-2:]
    last_two
    return (last_two,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Other helpful string methods

        Finally, here are some other helpful string methods. Feel free to try them out on your own strings by modifying the value of `sentence` below.
        """
    )
    return


@app.cell
def _():
    sentence = "  python is fun  "
    sentence
    return (sentence,)


@app.cell
def _(sentence):
    # Remove extra spaces
    sentence.strip()
    return


@app.cell
def _(sentence):
    # Split into a list of words
    sentence.split()
    return


@app.cell
def _(sentence):
    sentence.replace("fun", "awesome")
    return


@app.cell
def _():
    "123".isdigit(), "abc".isdigit()
    return


@app.cell
def _():
    "123".isalpha(), "abc".isalpha()
    return


@app.cell
def _():
    "Python3".isalnum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next steps

        For a full primer on strings, check out the [official documentation](https://docs.python.org/3/library/string.html).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

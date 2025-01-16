# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🎭 Strings

        Dive into the world of Python strings — where text comes to life! 

        ## Creating strings
        In Python, strings are containers for text. You can create them in two simple
        ways:

        ```python
        greeting = "Hello, Python!"  # using double quotes
        name = 'Alice'               # using single quotes
        ```

        Below is an example string.

        """
    )
    return


@app.cell
def _():
    text = "Python is amazing"
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
    mo.md("Use string methods and the `in` operator to find things in strings.")
    return


@app.cell
def _(text):
    text.find('is')
    return


@app.cell
def _(text):
    "Python" in text
    return


@app.cell
def _(text):
    "Javascript" in text
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Inserting values in strings
        
        Modern Python uses f-strings to insert values into strings. For example,
        check out how the next cell greets you (and notice the `f''''`)!

        Try changing the value of `my_name`, and watch how the greeting changes.
    """)
    return


@app.cell
def _():
    my_name = ''
    return (my_name,)


@app.cell
def _(my_name):
    f"Hello, {my_name}!"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Working with Parts of Strings
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

        Finally, here are some other helpful string methods. Feel free to try them out on your own strings!

        ```python
        sentence = "  python is fun  "

        # Remove extra spaces
        print(sentence.strip())        # "python is fun"

        # Split into a list of words
        print(sentence.split())        # ['python', 'is', 'fun']

        # Replace words
        print(sentence.replace('fun', 'awesome'))

        # Check what kind of text you have
        print("123".isdigit())        # True - only numbers?
        print("abc".isalpha())        # True - only letters?
        print("Python3".isalnum())    # True - letters or numbers?
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Your String Journey Begins!

    Next Steps:

    - Try combining different string methods

    - Practice with f-strings

    - Experiment with string slicing

    You're doing great! 🐍✨
    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


if __name__ == "__main__":
    app.run()

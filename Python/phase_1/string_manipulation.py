# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.12"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🎭 Python Strings

        Dive into the fascinating world of Python strings - where text becomes magic!

        ## Creating Your First Strings
        In Python, strings are like containers for text. You can create them in two simple ways:
        ```python
        greeting = "Hello, Python!"  # using double quotes
        name = 'Alice'              # using single quotes
        ```

        ## Essential String Operations
        Let us explore what you can do with strings:
        ```python
        text = "Python is amazing!"

        # Basic operations
        print(len(text))           # Count characters: 17
        print(text.upper())        # PYTHON IS AMAZING!
        print(text.lower())        # python is amazing!
        print(text.title())        # Python Is Amazing!

        # Finding things in strings
        print(text.find('is'))     # Find where 'is' starts: 7
        print('Python' in text)    # Check if 'Python' exists: True
        ```

        ## String Formatting Made Easy
        Modern Python uses f-strings - they are the easiest way to add variables to your text:
        ```python
        name = "Alice"
        age = 25
        message = f"Hi, I'm {name} and I'm {age} years old!"
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Working with Parts of Strings
        You can access any part of a string using its position (index):
        ```python
        text = "Python"

        first_letter = text[0]     # 'P'
        last_letter = text[-1]     # 'n'
        first_three = text[0:3]    # 'Pyt'
        last_two = text[-2:]       # 'on'
        ```

        ## Common String Methods You'll Love
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


@app.cell
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


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

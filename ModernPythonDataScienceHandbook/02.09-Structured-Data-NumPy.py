import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Structured Data: NumPy's Structured Arrays""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""While often our data can be well represented by a homogeneous array of values, sometimes this is not the case. This chapter demonstrates the use of NumPy's *structured arrays* and *record arrays*, which provide efficient storage for compound, heterogeneous data.  While the patterns shown here are useful for simple operations, scenarios like this often lend themselves to the use of Pandas ``DataFrame``s, which we'll explore in [Part 3](03.00-Introduction-to-Pandas.ipynb).""")
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Imagine that we have several categories of data on a number of people (say, name, age, and weight), and we'd like to store these values for use in a Python program.
        It would be possible to store these in three separate arrays:
        """
    )
    return


@app.cell
def _():
    name = ["Alice", "Bob", "Cathy", "Doug"]
    age = [25, 45, 37, 19]
    weight = [55.0, 85.5, 68.0, 61.5]
    return age, name, weight


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But this is a bit clumsy. There's nothing here that tells us that the three arrays are related; NumPy's structured arrays allow us to do this more naturally by using a single structure to store all of this data.

        Recall that previously we created a simple array using an expression like this:
        """
    )
    return


@app.cell
def _(np):
    x = np.zeros(4, dtype=int)
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can similarly create a structured array using a compound data type specification:""")
    return


@app.cell
def _(np):
    # Use a compound data type for structured arrays
    data = np.zeros(
        4,
        dtype={"names": ("name", "age", "weight"), "formats": ("U10", "i4", "f8")},
    )
    print(data.dtype)
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here `'U10'` translates to "Unicode string of maximum length 10," `'i4'` translates to "4-byte (i.e., 32-bit) integer," and `'f8'` translates to "8-byte (i.e., 64-bit) float."
        We'll discuss other options for these type codes in the following section.

        Now that we've created an empty container array, we can fill the array with our lists of values:
        """
    )
    return


@app.cell
def _(age, data, name, weight):
    data["name"] = name
    data["age"] = age
    data["weight"] = weight
    print(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As we had hoped, the data is now conveniently arranged in one structured array.

        The handy thing with structured arrays is that we can now refer to values either by index or by name:
        """
    )
    return


@app.cell
def _(data):
    # Get all names
    data["name"]
    return


@app.cell
def _(data):
    # Get first row of data
    data[0]
    return


@app.cell
def _(data):
    # Get the name from the last row
    data[-1]["name"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Using Boolean masking, we can even do some more sophisticated operations, such as filtering on age:""")
    return


@app.cell
def _(data):
    # Get names where age is under 30
    data[data["age"] < 30]["name"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you'd like to do any operations that are any more complicated than these, you should probably consider the Pandas package, covered in [Part 4](04.00-Introduction-To-Matplotlib.ipynb).
        As you'll see, Pandas provides a `DataFrame` object, which is a structure built on NumPy arrays that offers a variety of useful data manipulation functionality similar to what you've seen here, as well as much, much more.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exploring Structured Array Creation

        Structured array data types can be specified in a number of ways.
        Earlier, we saw the dictionary method:
        """
    )
    return


@app.cell
def _(np):
    np.dtype({"names": ("name", "age", "weight"), "formats": ("U10", "i4", "f8")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For clarity, numerical types can be specified using Python types or NumPy `dtype`s instead:""")
    return


@app.cell
def _(np):
    np.dtype(
        {
            "names": ("name", "age", "weight"),
            "formats": ((np.str_, 10), int, np.float32),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A compound type can also be specified as a list of tuples:""")
    return


@app.cell
def _(np):
    np.dtype([("name", "S10"), ("age", "i4"), ("weight", "f8")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If the names of the types do not matter to you, you can specify the types alone in a comma-separated string:""")
    return


@app.cell
def _(np):
    np.dtype("S10,i4,f8")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The shortened string format codes may not be immediately intuitive, but they are built on simple principles.
        The first (optional) character `<` or `>`, means "little endian" or "big endian," respectively, and specifies the ordering convention for significant bits.
        The next character specifies the type of data: characters, bytes, ints, floating points, and so on (see the table below).
        The last character or characters represent the size of the object in bytes.

        | Character    | Description           | Example                           |
        | ---------    | -----------           | -------                           | 
        | `'b'`        | Byte                  | `np.dtype('b')`                   |
        | `'i'`        | Signed integer        | `np.dtype('i4') == np.int32`      |
        | `'u'`        | Unsigned integer      | `np.dtype('u1') == np.uint8`      |
        | `'f'`        | Floating point        | `np.dtype('f8') == np.int64`      |
        | `'c'`        | Complex floating point| `np.dtype('c16') == np.complex128`|
        | `'S'`, `'a'` | String                | `np.dtype('S5')`                  |
        | `'U'`        | Unicode string        | `np.dtype('U') == np.str_`        |
        | `'V'`        | Raw data (void)       | `np.dtype('V') == np.void`        |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## More Advanced Compound Types

        It is possible to define even more advanced compound types.
        For example, you can create a type where each element contains an array or matrix of values.
        Here, we'll create a data type with a `mat` component consisting of a $3\times 3$ floating-point matrix:
        """
    )
    return


@app.cell
def _(np):
    tp = np.dtype([("id", "i8"), ("mat", "f8", (3, 3))])
    X = np.zeros(1, dtype=tp)
    print(X[0])
    print(X["mat"][0])
    return X, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now each element in the `X` array consists of an `id` and a $3\times 3$ matrix.
        Why would you use this rather than a simple multidimensional array, or perhaps a Python dictionary?
        One reason is that this NumPy `dtype` directly maps onto a C structure definition, so the buffer containing the array content can be accessed directly within an appropriately written C program.
        If you find yourself writing a Python interface to a legacy C or Fortran library that manipulates structured data, structured arrays can provide a powerful interface.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Record Arrays: Structured Arrays with a Twist

        NumPy also provides record arrays (instances of the `np.recarray` class), which are almost identical to the structured arrays just described, but with one additional feature: fields can be accessed as attributes rather than as dictionary keys.
        Recall that we previously accessed the ages in our sample dataset by writing:
        """
    )
    return


@app.cell
def _(data):
    data["age"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If we view our data as a record array instead, we can access this with slightly fewer keystrokes:""")
    return


@app.cell
def _(data, np):
    data_rec = data.view(np.recarray)
    data_rec.age
    return (data_rec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The downside is that for record arrays, there is some extra overhead involved in accessing the fields, even when using the same syntax:""")
    return


@app.cell
def _(data, data_rec):
    # magic command not supported in marimo; please file an issue to add support
    data["age"]
    data_rec["age"]
    data_rec.age
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Whether the more convenient notation is worth the (slight) overhead will depend on your own application.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## On to Pandas

        This chapter on structured and record arrays is purposely located at the end of this part of the book, because it leads so well into the next package we will cover: Pandas.
        Structured arrays can come in handy in certain situations, like when you're using NumPy arrays to map onto binary data formats in C, Fortran, or another language.
        But for day-to-day use of structured data, the Pandas package is a much better choice; we'll explore it in depth in the chapters that follow.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

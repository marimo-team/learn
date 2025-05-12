import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        **Notebook 2: NumPy for Operations and Arrays**

        In this notebook, you'll gain an understanding of NumPy's usage for basic arithmetic operations. Then, we will explore how to create matrices and how to apply these operations to them.
        """
    )
    return


@app.cell
def _():
    # If you haven't already, make sure to first install NumPy by running the command 'pip install numpy'

    import numpy as np
    return (np,)


@app.cell
def _(mo):
    mo.md(
        r"""
        **Basic Arithmetic**

        Let's begin with some basic operations. Run the cells below to see how NumPy can be used for operations and functions.
        """
    )
    return


@app.cell
def _(np):
    print(np.add(10,5))
    print(np.subtract(10,5))
    print(np.multiply(10,5))
    print(np.divide(10,5))
    return


@app.cell
def _(np):
    # It can also be used for power.
    print(np.power(2,3))
    return


@app.cell
def _(mo):
    mo.md(r"""Numpy can also be used for mathematical constants.""")
    return


@app.cell
def _(np):
    print(np.pi)
    print(np.sqrt(2))
    print(np.log(10))
    print(np.cos(np.pi / 2))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Below, try writing out this function with as few parenthesis as possible: 

        $$
        5 + \log(4) \times \pi^6 - \frac{5}{4}
        $$
        """
    )
    return


@app.cell
def _():
    # Your code here!
    # Hint: consider the order of PEMDAS.
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Now that you've seen how NumPy handles math, it's time to understand how arrays work.

        Arrays are basically 'super-powered' lists - they're more efficient, easier to use for math, and down the line, much better for handling big data.

        Let's bgin with a 1D array.
        """
    )
    return


@app.cell
def _():
    # In python, you might create a list like this.
    lst = [1,2,3,4,5]
    lst
    return (lst,)


@app.cell
def _(lst, np):
    # Using NumPy, we can create an array from this list. 
    arr = np.array(lst)
    return (arr,)


@app.cell
def _(mo):
    mo.md(r"""Let's learn more about the characteristics of arrays.""")
    return


@app.cell
def _(arr):
    print("Type: ", type(arr))
    print("Shape: ", arr.shape)
    print("Size: ", arr.size)
    print("Data type: ", arr.dtype)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        What does this mean?

        **Type** tells us the type of the array itself - it belongs to the class array.
        **shape** tells us the structure - here, it's a flat row with 5 values, or 5x1.
        **size** tells us the total amount of elements in the array. Lastly, in this case, **data type** tells us the data type of the entries within the array, in this case usually int64 or float64 (numbers).

        Below, try creating a 1D array with 10 numbers. Try printing its shape, size, and type.
        """
    )
    return


@app.cell
def _():
    # Your code here!
    return


if __name__ == "__main__":
    app.run()

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
        **Accessing and Transforming 2D Arrays**

        In this notebook, we'll focus on accessing individual values and slices of arrays, understanding how to work with 2D arrays, and learn the difference between copying and viewing arays.

        Let's get started!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Indexing in 1D Arrays**

        Let's begin with a simple 1D array and learn how to grab values based on their index position

        """
    )
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(np):
    # Run the code below. 
    arr = np.array([10, 20, 30, 40, 50])
    arr
    return (arr,)


@app.cell
def _(arr):
    # The following code indexes the first value and the last value of the array.
    print(arr[0])
    print(arr[-1])
    return


@app.cell
def _(mo):
    mo.md(r"""Simple enough - now let's look at slicing multiple values from the arrays.""")
    return


@app.cell
def _(arr):
    print(arr[:3]) # Gives us the first 3 values
    print(arr[1:4]) # Gives us the values from position 1-3
    print(arr[::2]) # Gives us every other value
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In general, you can use the format *[start:stop:step]*.

        Below, try to slice the array to only include the following values.
        """
    )
    return


@app.cell
def _(np):
    arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print(arr1["""Your code here!"""]) # Every third value, starting at position 0
    print(arr1["""Your code here!"""]) # The last two values
    print(arr1["""Your code here!"""]) # Values from index position 4 to 7
    return (arr1,)


@app.cell
def _(mo):
    mo.md(r"""Let's move on to indexing in 2D arrays. Take a look at the matrix (2D array) below.""")
    return


@app.cell
def _(np):
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrix
    return (matrix,)


@app.cell
def _(mo):
    mo.md(r"""Looks confusing at first? Remember, a matrix is essentially just many lists within a list. By cleverly slicing the matrix, we can access the values we need! Run the code below to see.""")
    return


@app.cell
def _(matrix):
    print("Top left:", matrix[0, 0]) 
    print("Middle:", matrix[1, 1])  
    print("Last column:", matrix[:, 2]) # ":" means 'take all' - helpful for grabbing entire rows or columns
    return


@app.cell
def _(mo):
    mo.md(r"""Very similar to 1D slicing. Below, try to slice the matrices yourself!""")
    return


@app.cell
def _():
    matrix1 = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]

    print(matrix1["""Your code here!"""]) # Grab the last element in the last row
    print(matrix1["""Your code here!"""]) # Grab the first element in the second column
    print(matrix1["""Your code here!"""]) # Grab the first and third elements in the second row
    return (matrix1,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Great! Now that we understand how to access certain elements, lets see how we can work with arrays even more.

        **Reshaping Arrays**

        Reshaping allows ou to change the layout of an array without changing the data itself. This is really useful when you need your data in a different format - for example, turning a long list into a grid.

        Run the code below to see an example of reshaping a 1D array into a 2D array.
        """
    )
    return


@app.cell
def _(np):
    arr2 = np.arange(12)
    print("Original: ", arr2)

    reshaped = arr2.reshape((3, 4))
    print("Reshaped: \n", reshaped)

    return arr2, reshaped


@app.cell
def _(mo):
    mo.md(
        r"""
        Notice how we turned a long, horizontal list into a 3 x 4 grid, or matrix. Recall that the first number, 3, represents the number of rows, while the second number, 4, represents the number of columns.

        You can also use -1 to let NumPy figure out the size for you. This is handy when you know how many rows you want, but are not sure how many columns.
        """
    )
    return


@app.cell
def _(arr2):
    reshaped_auto = arr2.reshape((4, -1))
    reshaped_auto
    return (reshaped_auto,)


@app.cell
def _(mo):
    mo.md(
        r"""
        **Flattening Arrays**

        On the contrary, flattening turns our multidimensional array into a single line, or 1D array. This comes to be in handy when, for example, feeding data into a machine learning model, saving or exporting data, or just attemping to simplify things. 

        Check out the example below.
        """
    )
    return


@app.cell
def _(matrix, np):
    matrix2 = np.array([[1, 2, 3], [4, 5, 6]])
    print("Original:\n", matrix)

    flat = matrix2.flatten()
    print("Flattened:", flat)
    return flat, matrix2


@app.cell
def _(mo):
    mo.md(r"""Helpful tip: *.flatten()* will always give you a copy of the data. """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Transposing Arrays**

        You might remember transposing if you took a linear algebra class. Here, transposing means 'flipping' the matrix - the rows become columns, and the columns become rows. 
        """
    )
    return


@app.cell
def _(np):
    arr3 = np.array([[1, 2, 3], [4, 5, 6]])
    print("Original:\n", arr3)

    transposed = arr3.T
    print("Transposed:\n", transposed)

    return arr3, transposed


@app.cell
def _(mo):
    mo.md(r"""You can use *.T* or *.transpose()* interchangebly.""")
    return


@app.cell
def _(mo):
    mo.md(r"""Let's do some practice below!""")
    return


app._unparsable_cell(
    r"""
    # First, create an array with numbers 1 to 16
    arr4 = np.arange()

    # Next, reshape it to 4x4
    arr4 = 

    # Transpose the matrix
    arr4 = 

    # Lastly, flatten the result
    arr4 = 
    """,
    name="_"
)


@app.cell
def _():
    # Challenge - can you transform this matrix back into the original?
    return


if __name__ == "__main__":
    app.run()

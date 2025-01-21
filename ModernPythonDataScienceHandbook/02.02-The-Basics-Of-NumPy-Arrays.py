import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# The Basics of NumPy Arrays""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Data manipulation in Python is nearly synonymous with NumPy array manipulation: even newer tools like Pandas ([Part 3](03.00-Introduction-to-Pandas.ipynb)) are built around the NumPy array.
        This chapter will present several examples of using NumPy array manipulation to access data and subarrays, and to split, reshape, and join the arrays.
        While the types of operations shown here may seem a bit dry and pedantic, they comprise the building blocks of many other examples used throughout the book.
        Get to know them well!

        We'll cover a few categories of basic array manipulations here:

        - *Attributes of arrays*: Determining the size, shape, memory consumption, and data types of arrays
        - *Indexing of arrays*: Getting and setting the values of individual array elements
        - *Slicing of arrays*: Getting and setting smaller subarrays within a larger array
        - *Reshaping of arrays*: Changing the shape of a given array
        - *Joining and splitting of arrays*: Combining multiple arrays into one, and splitting one array into many
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## NumPy Array Attributes""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        First let's discuss some useful array attributes.
        We'll start by defining random arrays of one, two, and three dimensions.
        We'll use NumPy's random number generator, which we will *seed* with a set value in order to ensure that the same random arrays are generated each time this code is run:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    rng = np.random.default_rng(seed=1701)  # seed for reproducibility

    x1 = rng.integers(10, size=6)  # one-dimensional array
    x2 = rng.integers(10, size=(3, 4))  # two-dimensional array
    x3 = rng.integers(10, size=(3, 4, 5))  # three-dimensional array
    return np, rng, x1, x2, x3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Each array has attributes including `ndim` (the number of dimensions), `shape` (the size of each dimension), `size` (the total size of the array), and `dtype` (the type of each element):""")
    return


@app.cell
def _(x3):
    print("x3 ndim: ", x3.ndim)
    print("x3 shape:", x3.shape)
    print("x3 size: ", x3.size)
    print("dtype:   ", x3.dtype)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For more discussion of data types, see [Understanding Data Types in Python](02.01-Understanding-Data-Types.ipynb).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Array Indexing: Accessing Single Elements""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you are familiar with Python's standard list indexing, indexing in NumPy will feel quite familiar.
        In a one-dimensional array, the $i^{th}$ value (counting from zero) can be accessed by specifying the desired index in square brackets, just as with Python lists:
        """
    )
    return


@app.cell
def _(x1):
    x1
    return


@app.cell
def _(x1):
    x1[0]
    return


@app.cell
def _(x1):
    x1[4]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To index from the end of the array, you can use negative indices:""")
    return


@app.cell
def _(x1):
    x1[-1]
    return


@app.cell
def _(x1):
    x1[-2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In a multidimensional array, items can be accessed using a comma-separated `(row, column)` tuple:""")
    return


@app.cell
def _(x2):
    x2
    return


@app.cell
def _(x2):
    x2[0, 0]
    return


@app.cell
def _(x2):
    x2[2, 0]
    return


@app.cell
def _(x2):
    x2[2, -1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Values can also be modified using any of the preceding index notation:""")
    return


@app.cell
def _(x2):
    x2[0, 0] = 12
    x2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Keep in mind that, unlike Python lists, NumPy arrays have a fixed type.
        This means, for example, that if you attempt to insert a floating-point value into an integer array, the value will be silently truncated. Don't be caught unaware by this behavior!
        """
    )
    return


@app.cell
def _(x1):
    x1[0] = 3.14159  # this will be truncated!
    x1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Array Slicing: Accessing Subarrays""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Just as we can use square brackets to access individual array elements, we can also use them to access subarrays with the *slice* notation, marked by the colon (`:`) character.
        The NumPy slicing syntax follows that of the standard Python list; to access a slice of an array `x`, use this:
        ``` python
        x[start:stop:step]
        ```
        If any of these are unspecified, they default to the values `start=0`, `stop=<size of dimension>`, `step=1`.
        Let's look at some examples of accessing subarrays in one dimension and in multiple dimensions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### One-Dimensional Subarrays

        Here are some examples of accessing elements in one-dimensional subarrays:
        """
    )
    return


@app.cell
def _(x1):
    x1
    return


@app.cell
def _(x1):
    x1[:3]  # first three elements
    return


@app.cell
def _(x1):
    x1[3:]  # elements after index 3
    return


@app.cell
def _(x1):
    x1[1:4]  # middle subarray
    return


@app.cell
def _(x1):
    x1[::2]  # every second element
    return


@app.cell
def _(x1):
    x1[1::2]  # every second element, starting at index 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A potentially confusing case is when the `step` value is negative.
        In this case, the defaults for `start` and `stop` are swapped.
        This becomes a convenient way to reverse an array:
        """
    )
    return


@app.cell
def _(x1):
    x1[::-1]  # all elements, reversed
    return


@app.cell
def _(x1):
    x1[4::-2]  # every second element from index 4, reversed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Multidimensional Subarrays

        Multidimensional slices work in the same way, with multiple slices separated by commas.
        For example:
        """
    )
    return


@app.cell
def _(x2):
    x2
    return


@app.cell
def _(x2):
    x2[:2, :3]  # first two rows & three columns
    return


@app.cell
def _(x2):
    x2[:3, ::2]  # three rows, every second column
    return


@app.cell
def _(x2):
    x2[::-1, ::-1]  # all rows & columns, reversed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Accessing array rows and columns

        One commonly needed routine is accessing single rows or columns of an array.
        This can be done by combining indexing and slicing, using an empty slice marked by a single colon (`:`):
        """
    )
    return


@app.cell
def _(x2):
    x2[:, 0]  # first column of x2
    return


@app.cell
def _(x2):
    x2[0, :]  # first row of x2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In the case of row access, the empty slice can be omitted for a more compact syntax:""")
    return


@app.cell
def _(x2):
    x2[0]  # equivalent to x2[0, :]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Subarrays as No-Copy Views

        Unlike Python list slices, NumPy array slices are returned as *views* rather than *copies* of the array data.
        Consider our two-dimensional array from before:
        """
    )
    return


@app.cell
def _(x2):
    print(x2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's extract a $2 \times 2$ subarray from this:""")
    return


@app.cell
def _(x2):
    x2_sub = x2[:2, :2]
    print(x2_sub)
    return (x2_sub,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now if we modify this subarray, we'll see that the original array is changed! Observe:""")
    return


@app.cell
def _(x2_sub):
    x2_sub[0, 0] = 99
    print(x2_sub)
    return


@app.cell
def _(x2):
    print(x2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Some users may find this surprising, but it can be advantageous: for example, when working with large datasets, we can access and process pieces of these datasets without the need to copy the underlying data buffer.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Creating Copies of Arrays

        Despite the nice features of array views, it is sometimes useful to instead explicitly copy the data within an array or a subarray. This can be most easily done with the `copy` method:
        """
    )
    return


@app.cell
def _(x2):
    x2_sub_copy = x2[:2, :2].copy()
    print(x2_sub_copy)
    return (x2_sub_copy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If we now modify this subarray, the original array is not touched:""")
    return


@app.cell
def _(x2_sub_copy):
    x2_sub_copy[0, 0] = 42
    print(x2_sub_copy)
    return


@app.cell
def _(x2):
    print(x2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Reshaping of Arrays

        Another useful type of operation is reshaping of arrays, which can be done with the `reshape` method.
        For example, if you want to put the numbers 1 through 9 in a $3 \times 3$ grid, you can do the following:
        """
    )
    return


@app.cell
def _(np):
    grid = np.arange(1, 10).reshape(3, 3)
    print(grid)
    return (grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Note that for this to work, the size of the initial array must match the size of the reshaped array, and in most cases the `reshape` method will return a no-copy view of the initial array.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A common reshaping operation is converting a one-dimensional array into a two-dimensional row or column matrix:""")
    return


@app.cell
def _(np):
    x = np.array([1, 2, 3])
    x.reshape((1, 3))  # row vector via reshape
    return (x,)


@app.cell
def _(x):
    x.reshape((3, 1))  # column vector via reshape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A convenient shorthand for this is to use `np.newaxis` in the slicing syntax:""")
    return


@app.cell
def _(np, x):
    x[np.newaxis, :]  # row vector via newaxis
    return


@app.cell
def _(np, x):
    x[:, np.newaxis]  # column vector via newaxis
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This is a pattern that we will utilize often throughout the remainder of the book.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Array Concatenation and Splitting

        All of the preceding routines worked on single arrays. NumPy also provides tools to combine multiple arrays into one, and to conversely split a single array into multiple arrays.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Concatenation of Arrays

        Concatenation, or joining of two arrays in NumPy, is primarily accomplished using the routines `np.concatenate`, `np.vstack`, and `np.hstack`.
        `np.concatenate` takes a tuple or list of arrays as its first argument, as you can see here:
        """
    )
    return


@app.cell
def _(np):
    x_1 = np.array([1, 2, 3])
    y = np.array([3, 2, 1])
    np.concatenate([x_1, y])
    return x_1, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can also concatenate more than two arrays at once:""")
    return


@app.cell
def _(np, x_1, y):
    z = np.array([99, 99, 99])
    print(np.concatenate([x_1, y, z]))
    return (z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And it can be used for two-dimensional arrays:""")
    return


@app.cell
def _(np):
    grid_1 = np.array([[1, 2, 3], [4, 5, 6]])
    return (grid_1,)


@app.cell
def _(grid_1, np):
    np.concatenate([grid_1, grid_1])
    return


@app.cell
def _(grid_1, np):
    np.concatenate([grid_1, grid_1], axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For working with arrays of mixed dimensions, it can be clearer to use the `np.vstack` (vertical stack) and `np.hstack` (horizontal stack) functions:""")
    return


@app.cell
def _(grid_1, np, x_1):
    np.vstack([x_1, grid_1])
    return


@app.cell
def _(grid_1, np):
    y_1 = np.array([[99], [99]])
    np.hstack([grid_1, y_1])
    return (y_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Similarly, for higher-dimensional arrays, `np.dstack` will stack arrays along the third axis.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Splitting of Arrays

        The opposite of concatenation is splitting, which is implemented by the functions `np.split`, `np.hsplit`, and `np.vsplit`.  For each of these, we can pass a list of indices giving the split points:
        """
    )
    return


@app.cell
def _(np):
    x_2 = [1, 2, 3, 99, 99, 3, 2, 1]
    x1_1, x2_1, x3_1 = np.split(x_2, [3, 5])
    print(x1_1, x2_1, x3_1)
    return x1_1, x2_1, x3_1, x_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that *N* split points leads to *N* + 1 subarrays.
        The related functions `np.hsplit` and `np.vsplit` are similar:
        """
    )
    return


@app.cell
def _(np):
    grid_2 = np.arange(16).reshape((4, 4))
    grid_2
    return (grid_2,)


@app.cell
def _(grid_2, np):
    upper, lower = np.vsplit(grid_2, [2])
    print(upper)
    print(lower)
    return lower, upper


@app.cell
def _(grid_2, np):
    left, right = np.hsplit(grid_2, [2])
    print(left)
    print(right)
    return left, right


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Similarly, for higher-dimensional arrays, `np.dsplit` will split arrays along the third axis.""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

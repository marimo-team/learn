import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Fancy Indexing""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The previous chapters discussed how to access and modify portions of arrays using simple indices (e.g., `arr[0]`), slices (e.g., `arr[:5]`), and Boolean masks (e.g., `arr[arr > 0]`).
        In this chapter, we'll look at another style of array indexing, known as *fancy* or *vectorized* indexing, in which we pass arrays of indices in place of single scalars.
        This allows us to very quickly access and modify complicated subsets of an array's values.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exploring Fancy Indexing

        Fancy indexing is conceptually simple: it means passing an array of indices to access multiple array elements at once.
        For example, consider the following array:
        """
    )
    return


@app.cell
def _():
    import numpy as np

    rng = np.random.default_rng(seed=1701)

    x = rng.integers(100, size=10)
    print(x)
    return np, rng, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Suppose we want to access three different elements. We could do it like this:""")
    return


@app.cell
def _(x):
    [x[3], x[7], x[2]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Alternatively, we can pass a single list or array of indices to obtain the same result:""")
    return


@app.cell
def _(x):
    ind = [3, 7, 4]
    x[ind]
    return (ind,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""When using arrays of indices, the shape of the result reflects the shape of the *index arrays* rather than the shape of the *array being indexed*:""")
    return


@app.cell
def _(np, x):
    ind_1 = np.array([[3, 7], [4, 5]])
    x[ind_1]
    return (ind_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Fancy indexing also works in multiple dimensions. Consider the following array:""")
    return


@app.cell
def _(np):
    X = np.arange(12).reshape((3, 4))
    X
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Like with standard indexing, the first index refers to the row, and the second to the column:""")
    return


@app.cell
def _(X, np):
    row = np.array([0, 1, 2])
    col = np.array([2, 1, 3])
    X[row, col]
    return col, row


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that the first value in the result is `X[0, 2]`, the second is `X[1, 1]`, and the third is `X[2, 3]`.
        The pairing of indices in fancy indexing follows all the broadcasting rules that were mentioned in [Computation on Arrays: Broadcasting](02.05-Computation-on-arrays-broadcasting.ipynb).
        So, for example, if we combine a column vector and a row vector within the indices, we get a two-dimensional result:
        """
    )
    return


@app.cell
def _(X, col, np, row):
    X[row[:, np.newaxis], col]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here, each row value is matched with each column vector, exactly as we saw in broadcasting of arithmetic operations.
        For example:
        """
    )
    return


@app.cell
def _(col, np, row):
    row[:, np.newaxis] * col
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""It is always important to remember with fancy indexing that the return value reflects the *broadcasted shape of the indices*, rather than the shape of the array being indexed.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Combined Indexing

        For even more powerful operations, fancy indexing can be combined with the other indexing schemes we've seen. For example, given the array `X`:
        """
    )
    return


@app.cell
def _(X):
    print(X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can combine fancy and simple indices:""")
    return


@app.cell
def _(X):
    X[2, [2, 0, 1]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also combine fancy indexing with slicing:""")
    return


@app.cell
def _(X):
    X[1:, [2, 0, 1]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And we can combine fancy indexing with masking:""")
    return


@app.cell
def _(X, np, row):
    mask = np.array([True, False, True, False])
    X[row[:, np.newaxis], mask]
    return (mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""All of these indexing options combined lead to a very flexible set of operations for efficiently accessing and modifying array values.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: Selecting Random Points

        One common use of fancy indexing is the selection of subsets of rows from a matrix.
        For example, we might have an $N$ by $D$ matrix representing $N$ points in $D$ dimensions, such as the following points drawn from a two-dimensional normal distribution:
        """
    )
    return


@app.cell
def _(rng):
    mean = [0, 0]
    cov = [[1, 2], [2, 5]]
    X_1 = rng.multivariate_normal(mean, cov, 100)
    X_1.shape
    return X_1, cov, mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Using the plotting tools we will discuss in [Introduction to Matplotlib](04.00-Introduction-To-Matplotlib.ipynb), we can visualize these points as a scatter plot (see the following figure):""")
    return


@app.cell
def _(X):
    import matplotlib.pyplot as plt
    # plt.style.use('seaborn-whitegrid')

    plt.scatter(X[:, 0], X[:, 1])
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's use fancy indexing to select 20 random points. We'll do this by first choosing 20 random indices with no repeats, and using these indices to select a portion of the original array:""")
    return


@app.cell
def _(X_1, np):
    indices = np.random.choice(X_1.shape[0], 20, replace=False)
    indices
    return (indices,)


@app.cell
def _(X_1, indices):
    selection = X_1[indices]
    selection.shape
    return (selection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now to see which points were selected, let's overplot large circles at the locations of the selected points (see the following figure):""")
    return


@app.cell
def _(X_1, plt, selection):
    plt.scatter(X_1[:, 0], X_1[:, 1], alpha=0.3)
    plt.scatter(
        selection[:, 0], selection[:, 1], facecolor="none", edgecolor="black", s=200
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This sort of strategy is often used to quickly partition datasets, as is often needed in train/test splitting for validation of statistical models (see [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb)), and in sampling approaches to answering statistical questions.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Modifying Values with Fancy Indexing

        Just as fancy indexing can be used to access parts of an array, it can also be used to modify parts of an array.
        For example, imagine we have an array of indices and we'd like to set the corresponding items in an array to some value:
        """
    )
    return


@app.cell
def _(np):
    x_1 = np.arange(10)
    i = np.array([2, 1, 8, 4])
    x_1[i] = 99
    print(x_1)
    return i, x_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can use any assignment-type operator for this. For example:""")
    return


@app.cell
def _(i, x_1):
    x_1[i] = x_1[i] - 10
    print(x_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Notice, though, that repeated indices with these operations can cause some potentially unexpected results. Consider the following:""")
    return


@app.cell
def _(np):
    x_2 = np.zeros(10)
    x_2[[0, 0]] = [4, 6]
    print(x_2)
    return (x_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Where did the 4 go? This operation first assigns `x[0] = 4`, followed by `x[0] = 6`.
        The result, of course, is that `x[0]` contains the value 6.

        Fair enough, but consider this operation:
        """
    )
    return


@app.cell
def _(x_2):
    i_1 = [2, 3, 3, 4, 4, 4]
    x_2[i_1] = x_2[i_1] + 1
    x_2
    return (i_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You might expect that `x[3]` would contain the value 2 and `x[4]` would contain the value 3, as this is how many times each index is repeated. Why is this not the case?
        Conceptually, this is because `x[i] += 1` is meant as a shorthand of `x[i] = x[i] + 1`. `x[i] + 1` is evaluated, and then the result is assigned to the indices in `x`.
        With this in mind, it is not the augmentation that happens multiple times, but the assignment, which leads to the rather nonintuitive results.

        So what if you want the other behavior where the operation is repeated? For this, you can use the `at` method of ufuncs and do the following:
        """
    )
    return


@app.cell
def _(i_1, np):
    x_3 = np.zeros(10)
    np.add.at(x_3, i_1, 1)
    print(x_3)
    return (x_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `at` method does an in-place application of the given operator at the specified indices (here, `i`) with the specified value (here, 1).
        Another method that is similar in spirit is the `reduceat` method of ufuncs, which you can read about in the [NumPy documentation](https://numpy.org/doc/stable/reference/ufuncs.html).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: Binning Data

        You could use these ideas to efficiently do custom binned computations on data.
        For example, imagine we have 100 values and would like to quickly find where they fall within an array of bins.
        We could compute this using `ufunc.at` like this:
        """
    )
    return


@app.cell
def _(np):
    rng_1 = np.random.default_rng(seed=1701)
    x_4 = rng_1.normal(size=100)
    bins = np.linspace(-5, 5, 20)
    counts = np.zeros_like(bins)
    i_2 = np.searchsorted(bins, x_4)
    np.add.at(counts, i_2, 1)
    return bins, counts, i_2, rng_1, x_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The counts now reflect the number of points within each bin—in other words, a histogram (see the following figure):""")
    return


@app.cell
def _(bins, counts, plt):
    # plot the results
    plt.plot(bins, counts, drawstyle="steps")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Of course, it would be inconvenient to have to do this each time you want to plot a histogram.
        This is why Matplotlib provides the `plt.hist` routine, which does the same in a single line:

        ```python
        plt.hist(x, bins, histtype='step');
        ```

        This function will create a nearly identical plot to the one just shown.
        To compute the binning, Matplotlib uses the `np.histogram` function, which does a very similar computation to what we did before. Let's compare the two here:
        """
    )
    return


@app.cell
def _(bins, np, x_4):
    print(f"NumPy histogram ({len(x_4)} points):")
    _counts, _edges = np.histogram(x_4, bins)

    print(f"Custom histogram ({len(x_4)} points):")
    np.add.at(_counts, np.searchsorted(bins, x_4), 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our own one-line algorithm is twice as fast as the optimized algorithm in NumPy! How can this be? If you dig into the `np.histogram` source code (you can do this in IPython by typing `np.histogram??`), you'll see that it's quite a bit more involved than the simple search-and-count that we've done; this is because NumPy's algorithm is more flexible, and particularly is designed for better performance when the number of data points becomes large:""")
    return


@app.cell
def _(bins, np, rng):
    _x = rng.normal(size=1000000)
    print(f"NumPy histogram ({len(_x)} points):")
    _counts, _edges = np.histogram(_x, bins)

    print(f"Custom histogram ({len(_x)} points):")
    np.add.at(_counts, np.searchsorted(bins, _x), 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        What this comparison shows is that algorithmic efficiency is almost never a simple question. An algorithm efficient for large datasets will not always be the best choice for small datasets, and vice versa (see [Big-O Notation](02.08-Sorting.ipynb#Big-O-Notation)).
        But the advantage of coding this algorithm yourself is that with an understanding of these basic methods, the sky is the limit: you're no longer constrained to built-in routines, but can create your own approaches to exploring the data.
        Key to efficiently using Python in data-intensive applications is not only knowing about general convenience routines like `np.histogram` and when they're appropriate, but also knowing how to make use of lower-level functionality when you need more pointed behavior.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

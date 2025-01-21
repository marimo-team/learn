import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Computation on NumPy Arrays: Universal Functions""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Up until now, we have been discussing some of the basic nuts and bolts of NumPy. In the next few chapters, we will dive into the reasons that NumPy is so important in the Python data science world: namely, because it provides an easy and flexible interface to optimize computation with arrays of data.

        Computation on NumPy arrays can be very fast, or it can be very slow.
        The key to making it fast is to use vectorized operations, generally implemented through NumPy's *universal functions* (ufuncs).
        This chapter motivates the need for NumPy's ufuncs, which can be used to make repeated calculations on array elements much more efficient.
        It then introduces many of the most common and useful arithmetic ufuncs available in the NumPy package.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Slowness of Loops

        Python's default implementation (known as CPython) does some operations very slowly.
        This is partly due to the dynamic, interpreted nature of the language; types are flexible, so sequences of operations cannot be compiled down to efficient machine code as in languages like C and Fortran.
        Recently there have been various attempts to address this weakness: well-known examples are the [PyPy project](http://pypy.org/), a just-in-time compiled implementation of Python; the [Cython project](http://cython.org), which converts Python code to compilable C code; and the [Numba project](http://numba.pydata.org/), which converts snippets of Python code to fast LLVM bytecode.
        Each of these has its strengths and weaknesses, but it is safe to say that none of the three approaches has yet surpassed the reach and popularity of the standard CPython engine.

        The relative sluggishness of Python generally manifests itself in situations where many small operations are being repeated; for instance, looping over arrays to operate on each element.
        For example, imagine we have an array of values and we'd like to compute the reciprocal of each.
        A straightforward approach might look like this:
        """
    )
    return


@app.cell
def _():
    import numpy as np

    rng = np.random.default_rng(seed=1701)


    def compute_reciprocals(values):
        output = np.empty(len(values))
        for i in range(len(values)):
            output[i] = 1.0 / values[i]
        return output


    values = rng.integers(1, 10, size=5)
    compute_reciprocals(values)
    return compute_reciprocals, np, rng, values


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This implementation probably feels fairly natural to someone from, say, a C or Java background.
        But if we measure the execution time of this code for a large input, we see that this operation is very slow—perhaps surprisingly so!
        We'll benchmark this with IPython's `%timeit` magic (discussed in [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb)):
        """
    )
    return


@app.cell
def _(compute_reciprocals, rng):
    big_array = rng.integers(1, 100, size=1000000)
    compute_reciprocals(big_array)
    return (big_array,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It takes several seconds to compute these million operations and to store the result!
        When even cell phones have processing speeds measured in gigaflops (i.e., billions of numerical operations per second), this seems almost absurdly slow.
        It turns out that the bottleneck here is not the operations themselves, but the type checking and function dispatches that CPython must do at each cycle of the loop.
        Each time the reciprocal is computed, Python first examines the object's type and does a dynamic lookup of the correct function to use for that type.
        If we were working in compiled code instead, this type specification would be known before the code executed and the result could be computed much more efficiently.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introducing Ufuncs

        For many types of operations, NumPy provides a convenient interface into just this kind of statically typed, compiled routine. This is known as a *vectorized* operation.
        For simple operations like the element-wise division here, vectorization is as simple as using Python arithmetic operators directly on the array object.
        This vectorized approach is designed to push the loop into the compiled layer that underlies NumPy, leading to much faster execution.

        Compare the results of the following two operations:
        """
    )
    return


@app.cell
def _(compute_reciprocals, values):
    print(compute_reciprocals(values))
    print(1.0 / values)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Looking at the execution time for our big array, we see that it completes orders of magnitude faster than the Python loop:""")
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %timeit (1.0 / big_array)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Vectorized operations in NumPy are implemented via ufuncs, whose main purpose is to quickly execute repeated operations on values in NumPy arrays.
        Ufuncs are extremely flexible—before we saw an operation between a scalar and an array, but we can also operate between two arrays:
        """
    )
    return


@app.cell
def _(np):
    np.arange(5) / np.arange(1, 6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And ufunc operations are not limited to one-dimensional arrays. They can act on multidimensional arrays as well:""")
    return


@app.cell
def _(np):
    x = np.arange(9).reshape((3, 3))
    2**x
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Computations using vectorization through ufuncs are nearly always more efficient than their counterparts implemented using Python loops, especially as the arrays grow in size.
        Any time you see such a loop in a NumPy script, you should consider whether it can be replaced with a vectorized expression.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exploring NumPy's Ufuncs

        Ufuncs exist in two flavors: *unary ufuncs*, which operate on a single input, and *binary ufuncs*, which operate on two inputs.
        We'll see examples of both these types of functions here.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Array Arithmetic

        NumPy's ufuncs feel very natural to use because they make use of Python's native arithmetic operators.
        The standard addition, subtraction, multiplication, and division can all be used:
        """
    )
    return


@app.cell
def _(np):
    x_1 = np.arange(4)
    print("x      =", x_1)
    print("x + 5  =", x_1 + 5)
    print("x - 5  =", x_1 - 5)
    print("x * 2  =", x_1 * 2)
    print("x / 2  =", x_1 / 2)
    print("x // 2 =", x_1 // 2)
    return (x_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""There is also a unary ufunc for negation, a `**` operator for exponentiation, and a `%` operator for modulus:""")
    return


@app.cell
def _(x_1):
    print("-x     = ", -x_1)
    print("x ** 2 = ", x_1**2)
    print("x % 2  = ", x_1 % 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In addition, these can be strung together however you wish, and the standard order of operations is respected:""")
    return


@app.cell
def _(x_1):
    -((0.5 * x_1 + 1) ** 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""All of these arithmetic operations are simply convenient wrappers around specific ufuncs built into NumPy. For example, the `+` operator is a wrapper for the `add` ufunc:""")
    return


@app.cell
def _(np, x_1):
    np.add(x_1, 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The following table lists the arithmetic operators implemented in NumPy:

        | Operator    | Equivalent ufunc  | Description                         |
        |-------------|-------------------|-------------------------------------|
        |`+`          |`np.add`           |Addition (e.g., `1 + 1 = 2`)         |
        |`-`          |`np.subtract`      |Subtraction (e.g., `3 - 2 = 1`)      |
        |`-`          |`np.negative`      |Unary negation (e.g., `-2`)          |
        |`*`          |`np.multiply`      |Multiplication (e.g., `2 * 3 = 6`)   |
        |`/`          |`np.divide`        |Division (e.g., `3 / 2 = 1.5`)       |
        |`//`         |`np.floor_divide`  |Floor division (e.g., `3 // 2 = 1`)  |
        |`**`         |`np.power`         |Exponentiation (e.g., `2 ** 3 = 8`)  |
        |`%`          |`np.mod`           |Modulus/remainder (e.g., `9 % 4 = 1`)|

        Additionally, there are Boolean/bitwise operators; we will explore these in [Comparisons, Masks, and Boolean Logic](02.06-Boolean-Arrays-and-Masks.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Absolute Value

        Just as NumPy understands Python's built-in arithmetic operators, it also understands Python's built-in absolute value function:
        """
    )
    return


@app.cell
def _(np):
    x_2 = np.array([-2, -1, 0, 1, 2])
    abs(x_2)
    return (x_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The corresponding NumPy ufunc is `np.absolute`, which is also available under the alias `np.abs`:""")
    return


@app.cell
def _(np, x_2):
    np.absolute(x_2)
    return


@app.cell
def _(np, x_2):
    np.abs(x_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This ufunc can also handle complex data, in which case it returns the magnitude:""")
    return


@app.cell
def _(np):
    x_3 = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
    np.abs(x_3)
    return (x_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Trigonometric Functions

        NumPy provides a large number of useful ufuncs, and some of the most useful for the data scientist are the trigonometric functions.
        We'll start by defining an array of angles:
        """
    )
    return


@app.cell
def _(np):
    theta = np.linspace(0, np.pi, 3)
    return (theta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we can compute some trigonometric functions on these values:""")
    return


@app.cell
def _(np, theta):
    print("theta      = ", theta)
    print("sin(theta) = ", np.sin(theta))
    print("cos(theta) = ", np.cos(theta))
    print("tan(theta) = ", np.tan(theta))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The values are computed to within machine precision, which is why values that should be zero do not always hit exactly zero.
        Inverse trigonometric functions are also available:
        """
    )
    return


@app.cell
def _(np):
    x_4 = [-1, 0, 1]
    print("x         = ", x_4)
    print("arcsin(x) = ", np.arcsin(x_4))
    print("arccos(x) = ", np.arccos(x_4))
    print("arctan(x) = ", np.arctan(x_4))
    return (x_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exponents and Logarithms

        Other common operations available in NumPy ufuncs are the exponentials:
        """
    )
    return


@app.cell
def _(np):
    x_5 = [1, 2, 3]
    print("x   =", x_5)
    print("e^x =", np.exp(x_5))
    print("2^x =", np.exp2(x_5))
    print("3^x =", np.power(3.0, x_5))
    return (x_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The inverse of the exponentials, the logarithms, are also available.
        The basic `np.log` gives the natural logarithm; if you prefer to compute the base-2 logarithm or the base-10 logarithm, these are available as well:
        """
    )
    return


@app.cell
def _(np):
    x_6 = [1, 2, 4, 10]
    print("x        =", x_6)
    print("ln(x)    =", np.log(x_6))
    print("log2(x)  =", np.log2(x_6))
    print("log10(x) =", np.log10(x_6))
    return (x_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""There are also some specialized versions that are useful for maintaining precision with very small input:""")
    return


@app.cell
def _(np):
    x_7 = [0, 0.001, 0.01, 0.1]
    print("exp(x) - 1 =", np.expm1(x_7))
    print("log(1 + x) =", np.log1p(x_7))
    return (x_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""When `x` is very small, these functions give more precise values than if the raw `np.log` or `np.exp` were to be used.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Specialized Ufuncs

        NumPy has many more ufuncs available, including for hyperbolic trigonometry, bitwise arithmetic, comparison operations, conversions from radians to degrees, rounding and remainders, and much more.
        A look through the NumPy documentation reveals a lot of interesting functionality.

        Another excellent source for more specialized ufuncs is the submodule `scipy.special`.
        If you want to compute some obscure mathematical function on your data, chances are it is implemented in `scipy.special`.
        There are far too many functions to list them all, but the following snippet shows a couple that might come up in a statistics context:
        """
    )
    return


@app.cell
def _():
    from scipy import special
    return (special,)


@app.cell
def _(special):
    x_8 = [1, 5, 10]
    print("gamma(x)     =", special.gamma(x_8))
    print("ln|gamma(x)| =", special.gammaln(x_8))
    print("beta(x, 2)   =", special.beta(x_8, 2))
    return (x_8,)


@app.cell
def _(np, special):
    x_9 = np.array([0, 0.3, 0.7, 1.0])
    print("erf(x)  =", special.erf(x_9))
    print("erfc(x) =", special.erfc(x_9))
    print("erfinv(x) =", special.erfinv(x_9))
    return (x_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are many, many more ufuncs available in both NumPy and `scipy.special`.
        Because the documentation of these packages is available online, a web search along the lines of "gamma function python" will generally find the relevant information.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Advanced Ufunc Features

        Many NumPy users make use of ufuncs without ever learning their full set of features.
        I'll outline a few specialized features of ufuncs here.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Specifying Output

        For large calculations, it is sometimes useful to be able to specify the array where the result of the calculation will be stored.
        For all ufuncs, this can be done using the `out` argument of the function:
        """
    )
    return


@app.cell
def _(np):
    x_10 = np.arange(5)
    y = np.empty(5)
    np.multiply(x_10, 10, out=y)
    print(y)
    return x_10, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This can even be used with array views. For example, we can write the results of a computation to every other element of a specified array:""")
    return


@app.cell
def _(np, x_10):
    y_1 = np.zeros(10)
    np.power(2, x_10, out=y_1[::2])
    print(y_1)
    return (y_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we had instead written `y[::2] = 2 ** x`, this would have resulted in the creation of a temporary array to hold the results of `2 ** x`, followed by a second operation copying those values into the `y` array.
        This doesn't make much of a difference for such a small computation, but for very large arrays the memory savings from careful use of the `out` argument can be significant.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Aggregations

        For binary ufuncs, aggregations can be computed directly from the object.
        For example, if we'd like to *reduce* an array with a particular operation, we can use the `reduce` method of any ufunc.
        A reduce repeatedly applies a given operation to the elements of an array until only a single result remains.

        For example, calling `reduce` on the `add` ufunc returns the sum of all elements in the array:
        """
    )
    return


@app.cell
def _(np):
    x_11 = np.arange(1, 6)
    np.add.reduce(x_11)
    return (x_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Similarly, calling `reduce` on the `multiply` ufunc results in the product of all array elements:""")
    return


@app.cell
def _(np, x_11):
    np.multiply.reduce(x_11)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If we'd like to store all the intermediate results of the computation, we can instead use `accumulate`:""")
    return


@app.cell
def _(np, x_11):
    np.add.accumulate(x_11)
    return


@app.cell
def _(np, x_11):
    np.multiply.accumulate(x_11)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Note that for these particular cases, there are dedicated NumPy functions to compute the results (`np.sum`, `np.prod`, `np.cumsum`, `np.cumprod`), which we'll explore in [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Outer Products

        Finally, any ufunc can compute the output of all pairs of two different inputs using the `outer` method.
        This allows you, in one line, to do things like create a multiplication table:
        """
    )
    return


@app.cell
def _(np):
    x_12 = np.arange(1, 6)
    np.multiply.outer(x_12, x_12)
    return (x_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `ufunc.at` and `ufunc.reduceat` methods are useful as well, and we will explore them in [Fancy Indexing](02.07-Fancy-Indexing.ipynb).

        We will also encounter the ability of ufuncs to operate between arrays of different shapes and sizes, a set of operations known as *broadcasting*.
        This subject is important enough that we will devote a whole chapter to it (see [Computation on Arrays: Broadcasting](02.05-Computation-on-arrays-broadcasting.ipynb)).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Ufuncs: Learning More""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        More information on universal functions (including the full list of available functions) can be found on the [NumPy](http://www.numpy.org) and [SciPy](http://www.scipy.org) documentation websites.

        Recall that you can also access information directly from within IPython by importing the packages and using IPython's tab completion and help (`?`) functionality, as described in [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

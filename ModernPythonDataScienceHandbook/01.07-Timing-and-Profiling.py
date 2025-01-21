import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Profiling and Timing Code
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the process of developing code and creating data processing pipelines, there are often trade-offs you can make between various implementations.
        Early in developing your algorithm, it can be counterproductive to worry about such things. As Donald Knuth famously quipped, "We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil."

        But once you have your code working, it can be useful to dig into its efficiency a bit.
        Sometimes it's useful to check the execution time of a given command or set of commands; other times it's useful to examine a multiline process and determine where the bottleneck lies in some complicated series of operations.
        IPython provides access to a wide array of functionality for this kind of timing and profiling of code.
        Here we'll discuss the following IPython magic commands:

        - `%time`: Time the execution of a single statement
        - `%timeit`: Time repeated execution of a single statement for more accuracy
        - `%prun`: Run code with the profiler
        - `%lprun`: Run code with the line-by-line profiler
        - `%memit`: Measure the memory use of a single statement
        - `%mprun`: Run code with the line-by-line memory profiler

        The last four commands are not bundled with IPython; to use them you'll need to get the `line_profiler` and `memory_profiler` extensions, which we will discuss in the following sections.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Timing Code Snippets: %timeit and %time

        We saw the `%timeit` line magic and `%%timeit` cell magic in the introduction to magic functions in [IPython Magic Commands](01.03-Magic-Commands.ipynb); these can be used to time the repeated execution of snippets of code:
        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %timeit sum(range(100))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that because this operation is so fast, `%timeit` automatically does a large number of repetitions.
        For slower commands, `%timeit` will automatically adjust and perform fewer repetitions:
        """
    )
    return


@app.cell
def _():
    total = 0
    for i in range(1000):
        for j in range(1000):
            total = total + i * (-1) ** j
    return i, j, total


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Sometimes repeating an operation is not the best option.
        For example, if we have a list that we'd like to sort, we might be misled by a repeated operation; sorting a pre-sorted list is much faster than sorting an unsorted list, so the repetition will skew the result:
        """
    )
    return


app._unparsable_cell(
    r"""
    import random
    L = [random.random() for i in range(100000)]
    %timeit L.sort()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For this, the `%time` magic function may be a better choice. It also is a good choice for longer-running commands, when short, system-related delays are unlikely to affect the result.
        Let's time the sorting of an unsorted and a presorted list:
        """
    )
    return


app._unparsable_cell(
    r"""
    L = [random.random() for i in range(100000)]
    print(\"sorting an unsorted list:\")
    %time L.sort()
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    print(\"sorting an already sorted list:\")
    %time L.sort()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice how much faster the presorted list is to sort, but notice also how much longer the timing takes with `%time` versus `%timeit`, even for the presorted list!
        This is a result of the fact that `%timeit` does some clever things under the hood to prevent system calls from interfering with the timing.
        For example, it prevents cleanup of unused Python objects (known as *garbage collection*) that might otherwise affect the timing.
        For this reason, `%timeit` results are usually noticeably faster than `%time` results.

        For `%time`, as with `%timeit`, using the `%%` cell magic syntax allows timing of multiline scripts:
        """
    )
    return


@app.cell
def _():
    total_1 = 0
    for i_1 in range(1000):
        for j_1 in range(1000):
            total_1 = total_1 + i_1 * (-1) ** j_1
    return i_1, j_1, total_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more information on `%time` and `%timeit`, as well as their available options, use the IPython help functionality (e.g., type `%time?` at the IPython prompt).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Profiling Full Scripts: %prun

        A program is made up of many single statements, and sometimes timing these statements in context is more important than timing them on their own.
        Python contains a built-in code profiler (which you can read about in the Python documentation), but IPython offers a much more convenient way to use this profiler, in the form of the magic function `%prun`.

        By way of example, we'll define a simple function that does some calculations:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        """
    )
    return


@app.cell
def _():
    def sum_of_lists(N):
        total = 0
        for i in range(5):
            L = [j ^ j >> i for j in range(N)]
            total = total + sum(L)
        return total
    return (sum_of_lists,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can call `%prun` with a function call to see the profiled results:
        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %prun # sum_of_lists(1000000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The result is a table that indicates, in order of total time on each function call, where the execution is spending the most time. In this case, the bulk of the execution time is in the list comprehension inside `sum_of_lists`.
        From here, we could start thinking about what changes we might make to improve the performance of the algorithm.

        For more information on `%prun`, as well as its available options, use the IPython help functionality (i.e., type `%prun?` at the IPython prompt).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Line-by-Line Profiling with %lprun

        The function-by-function profiling of `%prun` is useful, but sometimes it's more convenient to have a line-by-line profile report.
        This is not built into Python or IPython, but there is a `line_profiler` package available for installation that can do this.
        Start by using Python's packaging tool, `pip`, to install the `line_profiler` package:

        ```
        $ pip install line_profiler
        ```

        Next, you can use IPython to load the `line_profiler` IPython extension, offered as part of this package:
        """
    )
    return


@app.cell
def _():
    # '%load_ext line_profiler' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now the `%lprun` command will do a line-by-line profiling of any function. In this case, we need to tell it explicitly which functions we're interested in profiling:
        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %lprun # -f sum_of_lists sum_of_lists(5000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The information at the top gives us the key to reading the results: the time is reported in microseconds, and we can see where the program is spending the most time.
        At this point, we may be able to use this information to modify aspects of the script and make it perform better for our desired use case.

        For more information on `%lprun`, as well as its available options, use the IPython help functionality (i.e., type `%lprun?` at the IPython prompt).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Profiling Memory Use: %memit and %mprun

        Another aspect of profiling is the amount of memory an operation uses.
        This can be evaluated with another IPython extension, the `memory_profiler`.
        As with the `line_profiler`, we start by `pip`-installing the extension:

        ```
        $ pip install memory_profiler
        ```

        Then we can use IPython to load it:
        """
    )
    return


@app.cell
def _():
    # '%load_ext memory_profiler' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The memory profiler extension contains two useful magic functions: `%memit` (which offers a memory-measuring equivalent of `%timeit`) and `%mprun` (which offers a memory-measuring equivalent of `%lprun`).
        The `%memit` magic function can be used rather simply:
        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %memit # sum_of_lists(1000000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that this function uses about 140 MB of memory.

        For a line-by-line description of memory use, we can use the `%mprun` magic function.
        Unfortunately, this works only for functions defined in separate modules rather than the notebook itself, so we'll start by using the `%%file` cell magic to create a simple module called `mprun_demo.py`, which contains our `sum_of_lists` function, with one addition that will make our memory profiling results more clear:
        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%file mprun_demo.py
    # def sum_of_lists(N):
    #     total = 0
    #     for i in range(5):
    #         L = [j ^ (j >> i) for j in range(N)]
    #         total += sum(L)
    #         del L # remove reference to L
    #     return total
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can now import the new version of this function and run the memory line profiler:
        """
    )
    return


app._unparsable_cell(
    r"""
    from mprun_demo import sum_of_lists
    %mprun -f sum_of_lists sum_of_lists(1000000)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here, the `Increment` column tells us how much each line affects the total memory budget: observe that when we create and delete the list `L`, we are adding about 30 MB of memory usage.
        This is on top of the background memory usage from the Python interpreter itself.

        For more information on `%memit` and `%mprun`, as well as their available options, use the IPython help functionality (e.g., type `%memit?` at the IPython prompt).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

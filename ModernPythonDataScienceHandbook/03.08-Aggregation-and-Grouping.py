import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Aggregation and Grouping
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A fundamental piece of many data analysis tasks is efficient summarization: computing aggregations like `sum`, `mean`, `median`, `min`, and `max`, in which a single number summarizes aspects of a potentially large dataset.
        In this chapter, we'll explore aggregations in Pandas, from simple operations akin to what we've seen on NumPy arrays to more sophisticated operations based on the concept of a `groupby`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For convenience, we'll use the same `display` magic function that we used in the previous chapters:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    class display(object):
        """Display HTML representation of multiple objects"""
        template = """<div style="float: left; padding: 10px;">
        <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
        </div>"""
        def __init__(self, *args):
            self.args = args
            
        def _repr_html_(self):
            return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                             for a in self.args)
        
        def __repr__(self):
            return '\n\n'.join(a + '\n' + repr(eval(a))
                               for a in self.args)
    return display, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Planets Data

        Here we will use the Planets dataset, available via the [Seaborn package](http://seaborn.pydata.org/) (see [Visualization With Seaborn](04.14-Visualization-With-Seaborn.ipynb)).
        It gives information on planets that astronomers have discovered around other stars (known as *extrasolar planets*, or *exoplanets* for short). It can be downloaded with a simple Seaborn command:
        """
    )
    return


@app.cell
def _():
    import seaborn as sns
    planets = sns.load_dataset('planets')
    planets.shape
    return planets, sns


@app.cell
def _(planets):
    planets.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This has some details on the 1,000+ extrasolar planets discovered up to 2014.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Simple Aggregation in Pandas
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In ["Aggregations: Min, Max, and Everything In Between"](02.04-Computation-on-arrays-aggregates.ipynb), we explored some of the data aggregations available for NumPy arrays.
        As with a one-dimensional NumPy array, for a Pandas ``Series`` the aggregates return a single value:
        """
    )
    return


@app.cell
def _(np, pd):
    rng = np.random.RandomState(42)
    ser = pd.Series(rng.rand(5))
    ser
    return rng, ser


@app.cell
def _(ser):
    ser.sum()
    return


@app.cell
def _(ser):
    ser.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For a `DataFrame`, by default the aggregates return results within each column:
        """
    )
    return


@app.cell
def _(pd, rng):
    df = pd.DataFrame({'A': rng.rand(5),
                       'B': rng.rand(5)})
    df
    return (df,)


@app.cell
def _(df):
    df.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By specifying the `axis` argument, you can instead aggregate within each row:
        """
    )
    return


@app.cell
def _(df):
    df.mean(axis='columns')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Pandas `Series` and `DataFrame` objects include all of the common aggregates mentioned in [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb); in addition, there is a convenience method, `describe`, that computes several common aggregates for each column and returns the result.
        Let's use this on the Planets data, for now dropping rows with missing values:
        """
    )
    return


@app.cell
def _(planets):
    planets.dropna().describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This method helps us understand the overall properties of a dataset.
        For example, we see in the `year` column that although exoplanets were discovered as far back as 1989, half of all planets in the dataset were not discovered until 2010 or after.
        This is largely thanks to the *Kepler* mission, which aimed to find eclipsing planets around other stars using a specially designed space telescope.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The following table summarizes some other built-in Pandas aggregations:

        | Aggregation              | Returns                         |
        |--------------------------|---------------------------------|
        | ``count``                | Total number of items           |
        | ``first``, ``last``      | First and last item             |
        | ``mean``, ``median``     | Mean and median                 |
        | ``min``, ``max``         | Minimum and maximum             |
        | ``std``, ``var``         | Standard deviation and variance |
        | ``mad``                  | Mean absolute deviation         |
        | ``prod``                 | Product of all items            |
        | ``sum``                  | Sum of all items                |

        These are all methods of `DataFrame` and `Series` objects.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To go deeper into the data, however, simple aggregates are often not enough.
        The next level of data summarization is the `groupby` operation, which allows you to quickly and efficiently compute aggregates on subsets of data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## groupby: Split, Apply, Combine

        Simple aggregations can give you a flavor of your dataset, but often we would prefer to aggregate conditionally on some label or index: this is implemented in the so-called `groupby` operation.
        The name "group by" comes from a command in the SQL database language, but it is perhaps more illuminative to think of it in the terms first coined by Hadley Wickham of Rstats fame: *split, apply, combine*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Split, Apply, Combine

        A canonical example of this split-apply-combine operation, where the "apply" is a summation aggregation, is illustrated in this figure:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ![](images/03.08-split-apply-combine.png)

        ([figure source in Appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb#Split-Apply-Combine))
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This illustrates what the `groupby` operation accomplishes:

        - The *split* step involves breaking up and grouping a `DataFrame` depending on the value of the specified key.
        - The *apply* step involves computing some function, usually an aggregate, transformation, or filtering, within the individual groups.
        - The *combine* step merges the results of these operations into an output array.

        While this could certainly be done manually using some combination of the masking, aggregation, and merging commands covered earlier, an important realization is that *the intermediate splits do not need to be explicitly instantiated*. Rather, the `groupby` can (often) do this in a single pass over the data, updating the sum, mean, count, min, or other aggregate for each group along the way.
        The power of the `groupby` is that it abstracts away these steps: the user need not think about *how* the computation is done under the hood, but rather can think about the *operation as a whole*.

        As a concrete example, let's take a look at using Pandas for the computation shown in the following figure.
        We'll start by creating the input `DataFrame`:
        """
    )
    return


@app.cell
def _(pd):
    df_1 = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data': range(6)}, columns=['key', 'data'])
    df_1
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The most basic split-apply-combine operation can be computed with the `groupby` method of the `DataFrame`, passing the name of the desired key column:
        """
    )
    return


@app.cell
def _(df_1):
    df_1.groupby('key')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that what is returned is a `DataFrameGroupBy` object, not a set of `DataFrame` objects.
        This object is where the magic is: you can think of it as a special view of the `DataFrame`, which is poised to dig into the groups but does no actual computation until the aggregation is applied.
        This "lazy evaluation" approach means that common aggregates can be implemented efficiently in a way that is almost transparent to the user.

        To produce a result, we can apply an aggregate to this `DataFrameGroupBy` object, which will perform the appropriate apply/combine steps to produce the desired result:
        """
    )
    return


@app.cell
def _(df_1):
    df_1.groupby('key').sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `sum` method is just one possibility here; you can apply most Pandas or NumPy aggregation functions, as well as most `DataFrame` operations, as you will see in the following discussion.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The GroupBy Object

        The `GroupBy` object is a flexible abstraction: in many ways, it can be treated as simply a collection of ``DataFrame``s, though it is doing more sophisticated things under the hood. Let's see some examples using the Planets data.

        Perhaps the most important operations made available by a `GroupBy` are *aggregate*, *filter*, *transform*, and *apply*.
        We'll discuss each of these more fully in the next section, but before that let's take a look at some of the other functionality that can be used with the basic `GroupBy` operation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Column indexing

        The `GroupBy` object supports column indexing in the same way as the `DataFrame`, and returns a modified `GroupBy` object.
        For example:
        """
    )
    return


@app.cell
def _(planets):
    planets.groupby('method')
    return


@app.cell
def _(planets):
    planets.groupby('method')['orbital_period']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here we've selected a particular `Series` group from the original `DataFrame` group by reference to its column name.
        As with the `GroupBy` object, no computation is done until we call some aggregate on the object:
        """
    )
    return


@app.cell
def _(planets):
    planets.groupby('method')['orbital_period'].median()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This gives an idea of the general scale of orbital periods (in days) that each method is sensitive to.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Iteration over groups

        The `GroupBy` object supports direct iteration over the groups, returning each group as a `Series` or `DataFrame`:
        """
    )
    return


@app.cell
def _(planets):
    for (method, group) in planets.groupby('method'):
        print("{0:30s} shape={1}".format(method, group.shape))
    return group, method


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This can be useful for manual inspection of groups for the sake of debugging, but it is often much faster to use the built-in `apply` functionality, which we will discuss momentarily.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Dispatch methods

        Through some Python class magic, any method not explicitly implemented by the `GroupBy` object will be passed through and called on the groups, whether they are `DataFrame` or `Series` objects.
        For example, using the `describe` method is equivalent to calling `describe` on the `DataFrame` representing each group:
        """
    )
    return


@app.cell
def _(planets):
    planets.groupby('method')['year'].describe().unstack()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Looking at this table helps us to better understand the data: for example, the vast majority of planets until 2014 were discovered by the Radial Velocity and Transit methods, though the latter method became common more recently.
        The newest methods seem to be Transit Timing Variation and Orbital Brightness Modulation, which were not used to discover a new planet until 2011.

        Notice that these dispatch methods are applied *to each individual group*, and the results are then combined within `GroupBy` and returned.
        Again, any valid `DataFrame`/`Series` method can be called in a similar manner on the corresponding `GroupBy` object.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Aggregate, Filter, Transform, Apply

        The preceding discussion focused on aggregation for the combine operation, but there are more options available.
        In particular, `GroupBy` objects have `aggregate`, `filter`, `transform`, and `apply` methods that efficiently implement a variety of useful operations before combining the grouped data.

        For the purpose of the following subsections, we'll use this ``DataFrame``:
        """
    )
    return


@app.cell
def _(np, pd):
    rng_1 = np.random.RandomState(0)
    df_2 = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data1': range(6), 'data2': rng_1.randint(0, 10, 6)}, columns=['key', 'data1', 'data2'])
    df_2
    return df_2, rng_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Aggregation

        You're now familiar with `GroupBy` aggregations with `sum`, `median`, and the like, but the `aggregate` method allows for even more flexibility.
        It can take a string, a function, or a list thereof, and compute all the aggregates at once.
        Here is a quick example combining all of these:
        """
    )
    return


@app.cell
def _(df_2, np):
    df_2.groupby('key').aggregate(['min', np.median, max])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Another common pattern is to pass a dictionary mapping column names to operations to be applied on that column:
        """
    )
    return


@app.cell
def _(df_2):
    df_2.groupby('key').aggregate({'data1': 'min', 'data2': 'max'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Filtering

        A filtering operation allows you to drop data based on the group properties.
        For example, we might want to keep all groups in which the standard deviation is larger than some critical value:
        """
    )
    return


@app.cell
def _(display):
    def filter_func(x):
        return x['data2'].std() > 4

    display('df', "df.groupby('key').std()",
            "df.groupby('key').filter(filter_func)")
    return (filter_func,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The filter function should return a Boolean value specifying whether the group passes the filtering. Here, because group A does not have a standard deviation greater than 4, it is dropped from the result.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Transformation

        While aggregation must return a reduced version of the data, transformation can return some transformed version of the full data to recombine.
        For such a transformation, the output is the same shape as the input.
        A common example is to center the data by subtracting the group-wise mean:
        """
    )
    return


@app.cell
def _(df_2):
    def center(x):
        return x - x.mean()
    df_2.groupby('key').transform(center)
    return (center,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The apply method

        The `apply` method lets you apply an arbitrary function to the group results.
        The function should take a `DataFrame` and returns either a Pandas object (e.g., `DataFrame`, `Series`) or a scalar; the behavior of the combine step will be tailored to the type of output returned.

        For example, here is an `apply` operation that normalizes the first column by the sum of the second:
        """
    )
    return


@app.cell
def _(df_2):
    def norm_by_data2(x):
        x['data1'] = x['data1'] / x['data2'].sum()
        return x
    df_2.groupby('key').apply(norm_by_data2)
    return (norm_by_data2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        `apply` within a `GroupBy` is flexible: the only criterion is that the function takes a `DataFrame` and returns a Pandas object or scalar. What you do in between is up to you!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Specifying the Split Key

        In the simple examples presented before, we split the `DataFrame` on a single column name.
        This is just one of many options by which the groups can be defined, and we'll go through some other options for group specification here.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### A list, array, series, or index providing the grouping keys

        The key can be any series or list with a length matching that of the `DataFrame`. For example:
        """
    )
    return


@app.cell
def _(df_2):
    L = [0, 1, 0, 1, 2, 0]
    df_2.groupby(L).sum()
    return (L,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Of course, this means there's another, more verbose way of accomplishing the `df.groupby('key')` from before:
        """
    )
    return


@app.cell
def _(df_2):
    df_2.groupby(df_2['key']).sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### A dictionary or series mapping index to group

        Another method is to provide a dictionary that maps index values to the group keys:
        """
    )
    return


@app.cell
def _(df_2, display):
    df2 = df_2.set_index('key')
    mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
    display('df2', 'df2.groupby(mapping).sum()')
    return df2, mapping


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Any Python function

        Similar to mapping, you can pass any Python function that will input the index value and output the group:
        """
    )
    return


@app.cell
def _(df2):
    df2.groupby(str.lower).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### A list of valid keys

        Further, any of the preceding key choices can be combined to group on a multi-index:
        """
    )
    return


@app.cell
def _(df2, mapping):
    df2.groupby([str.lower, mapping]).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Grouping Example

        As an example of this, in a few lines of Python code we can put all these together and count discovered planets by method and by decade:
        """
    )
    return


@app.cell
def _(planets):
    decade = 10 * (planets['year'] // 10)
    decade = decade.astype(str) + 's'
    decade.name = 'decade'
    planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
    return (decade,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This shows the power of combining many of the operations we've discussed up to this point when looking at realistic datasets: we quickly gain a coarse understanding of when and how extrasolar planets were detected in the years after the first discovery.

        I would suggest digging into these few lines of code and evaluating the individual steps to make sure you understand exactly what they are doing to the result.
        It's certainly a somewhat complicated example, but understanding these pieces will give you the means to similarly explore your own data.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

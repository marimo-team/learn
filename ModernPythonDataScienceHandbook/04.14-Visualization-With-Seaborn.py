import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Visualization with Seaborn
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Matplotlib has been at the core of scientific visualization in Python for decades, but even avid users will admit it often leaves much to be desired.
        There are several complaints about Matplotlib that often come up:

        - A common early complaint, which is now outdated: prior to version 2.0, Matplotlib's color and style defaults were at times poor and looked dated.
        - Matplotlib's API is relatively low-level. Doing sophisticated statistical visualization is possible, but often requires a *lot* of boilerplate code.
        - Matplotlib predated Pandas by more than a decade, and thus is not designed for use with Pandas `DataFrame` objects. In order to visualize data from a `DataFrame`, you must extract each `Series` and often concatenate them together into the right format. It would be nicer to have a plotting library that can intelligently use the `DataFrame` labels in a plot.

        An answer to these problems is [Seaborn](http://seaborn.pydata.org/). Seaborn provides an API on top of Matplotlib that offers sane choices for plot style and color defaults, defines simple high-level functions for common statistical plot types, and integrates with the functionality provided by Pandas.

        To be fair, the Matplotlib team has adapted to the changing landscape: it added the `plt.style` tools discussed in [Customizing Matplotlib: Configurations and Style Sheets](04.11-Settings-and-Stylesheets.ipynb), and Matplotlib is starting to handle Pandas data more seamlessly.
        But for all the reasons just discussed, Seaborn remains a useful add-on.

        By convention, Seaborn is often imported as `sns`:
        """
    )
    return


@app.cell
def _():
    # "%matplotlib inline\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport numpy as np\nimport pandas as pd\n\nsns.set()  # seaborn's method to set its chart style" command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exploring Seaborn Plots

        The main idea of Seaborn is that it provides high-level commands to create a variety of plot types useful for statistical data exploration, and even some statistical model fitting.

        Let's take a look at a few of the datasets and plot types available in Seaborn. Note that all of the following *could* be done using raw Matplotlib commands (this is, in fact, what Seaborn does under the hood), but the Seaborn API is much more convenient.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Histograms, KDE, and Densities

        Often in statistical data visualization, all you want is to plot histograms and joint distributions of variables.
        We have seen that this is relatively straightforward in Matplotlib (see the following figure):
        """
    )
    return


@app.cell
def _(np, pd, plt):
    data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x', 'y'])

    for col in 'xy':
        plt.hist(data[col], density=True, alpha=0.5)
    return col, data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Rather than just providing a histogram as a visual output, we can get a smooth estimate of the distribution using kernel density estimation (introduced in [Density and Contour Plots](04.04-Density-and-Contour-Plots.ipynb)), which Seaborn does with ``sns.kdeplot`` (see the following figure):
        """
    )
    return


@app.cell
def _(data, sns):
    sns.kdeplot(data=data, shade=True);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we pass `x` and `y` columns to `kdeplot`, we instead get a two-dimensional visualization of the joint density (see the following figure):
        """
    )
    return


@app.cell
def _(data, sns):
    sns.kdeplot(data=data, x='x', y='y');
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can see the joint distribution and the marginal distributions together using `sns.jointplot`, which we'll explore further later in this chapter.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Pair Plots

        When you generalize joint plots to datasets of larger dimensions, you end up with *pair plots*. These are very useful for exploring correlations between multidimensional data, when you'd like to plot all pairs of values against each other.

        We'll demo this with the well-known Iris dataset, which lists measurements of petals and sepals of three Iris species:
        """
    )
    return


@app.cell
def _(sns):
    iris = sns.load_dataset("iris")
    iris.head()
    return (iris,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Visualizing the multidimensional relationships among the samples is as easy as calling ``sns.pairplot`` (see the following figure):
        """
    )
    return


@app.cell
def _(iris, sns):
    sns.pairplot(iris, hue='species', height=2.5);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Faceted Histograms

        Sometimes the best way to view data is via histograms of subsets, as shown in the following figure. Seaborn's `FacetGrid` makes this simple.
        We'll take a look at some data that shows the amount that restaurant staff receive in tips based on various indicator data:[^1]

        [^1]: The restaurant staff data used in this section divides employees into two sexes: female and male. Biological sex
        isn’t binary, but the following discussion and visualizations are limited by this data.
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
def _(sns):
    tips = sns.load_dataset('tips')
    tips.head()
    return (tips,)


@app.cell
def _(np, plt, sns, tips):
    tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

    grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
    grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15));
    return (grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The faceted chart gives us some quick insights into the dataset: for example, we see that it contains far more data on male servers during the dinner hour than other categories, and typical tip amounts appear to range from approximately 10% to 20%, with some outliers on either end.

        ### Categorical Plots

        Categorical plots can be useful for this kind of visualization as well. These allow you to view the distribution of a parameter within bins defined by any other parameter, as shown in the following figure:
        """
    )
    return


@app.cell
def _(sns, tips):
    with sns.axes_style(style='ticks'):
        _g = sns.catplot(x='day', y='total_bill', hue='sex', data=tips, kind='box')
        _g.set_axis_labels('Day', 'Total Bill')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Joint Distributions

        Similar to the pair plot we saw earlier, we can use `sns.jointplot` to show the joint distribution between different datasets, along with the associated marginal distributions (see the following figure):
        """
    )
    return


@app.cell
def _(sns, tips):
    with sns.axes_style('white'):
        sns.jointplot(x="total_bill", y="tip", data=tips, kind='hex')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The joint plot can even do some automatic kernel density estimation and regression, as shown in the following figure:
        """
    )
    return


@app.cell
def _(sns, tips):
    sns.jointplot(x="total_bill", y="tip", data=tips, kind='reg');
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Bar Plots

        Time series can be plotted using `sns.factorplot`. In the following example, we'll use the Planets dataset that we first saw in [Aggregation and Grouping](03.08-Aggregation-and-Grouping.ipynb); see the following figure for the result:
        """
    )
    return


@app.cell
def _(sns):
    planets = sns.load_dataset('planets')
    planets.head()
    return (planets,)


@app.cell
def _(planets, sns):
    with sns.axes_style('white'):
        _g = sns.catplot(x='year', data=planets, aspect=2, kind='count', color='steelblue')
        _g.set_xticklabels(step=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can learn more by looking at the *method* of discovery of each of these planets (see the following figure):
        """
    )
    return


@app.cell
def _(planets, sns):
    with sns.axes_style('white'):
        _g = sns.catplot(x='year', data=planets, aspect=4.0, kind='count', hue='method', order=range(2001, 2015))
        _g.set_ylabels('Number of Planets Discovered')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more information on plotting with Seaborn, see the [Seaborn documentation](http://seaborn.pydata.org/), and particularly the [example gallery](https://seaborn.pydata.org/examples/index.html).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: Exploring Marathon Finishing Times

        Here we'll look at using Seaborn to help visualize and understand finishing results from a marathon.
        I've scraped the data from sources on the web, aggregated it and removed any identifying information, and put it on GitHub, where it can be downloaded
        (if you are interested in using Python for web scraping, I would recommend [*Web Scraping with Python*](http://shop.oreilly.com/product/0636920034391.do) by Ryan Mitchell, also from O'Reilly).
        We will start by downloading the data and loading it into Pandas:[^2]

        [^2]: The marathon data used in this section divides runners into two genders: men and women. While gender is a
        spectrum, the following discussion and visualizations use this binary because they depend on the data.
        """
    )
    return


@app.cell
def _():
    # url = ('https://raw.githubusercontent.com/jakevdp/'
    #        'marathon-data/master/marathon-data.csv')
    # !cd data && curl -O {url}
    return


@app.cell
def _(pd):
    data_1 = pd.read_csv('data/marathon-data.csv')
    data_1.head()
    return (data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that Pandas loaded the time columns as Python strings (type `object`); we can see this by looking at the `dtypes` attribute of the `DataFrame`:
        """
    )
    return


@app.cell
def _(data_1):
    data_1.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's fix this by providing a converter for the times:
        """
    )
    return


@app.cell
def _(pd):
    import datetime

    def convert_time(s):
        h, m, s = map(int, s.split(':'))
        return datetime.timedelta(hours=h, minutes=m, seconds=s)
    data_2 = pd.read_csv('data/marathon-data.csv', converters={'split': convert_time, 'final': convert_time})
    data_2.head()
    return convert_time, data_2, datetime


@app.cell
def _(data_2):
    data_2.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        That will make it easier to manipulate the temporal data. For the purpose of our Seaborn plotting utilities, let's next add columns that give the times in seconds:
        """
    )
    return


@app.cell
def _(data_2):
    data_2['split_sec'] = data_2['split'].view(int) / 1000000000.0
    data_2['final_sec'] = data_2['final'].view(int) / 1000000000.0
    data_2.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To get an idea of what the data looks like, we can plot a `jointplot` over the data; the following figure shows the result:
        """
    )
    return


@app.cell
def _(data_2, np, sns):
    with sns.axes_style('white'):
        _g = sns.jointplot(x='split_sec', y='final_sec', data=data_2, kind='hex')
        _g.ax_joint.plot(np.linspace(4000, 16000), np.linspace(8000, 32000), ':k')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The dotted line shows where someone's time would lie if they ran the marathon at a perfectly steady pace. The fact that the distribution lies above this indicates (as you might expect) that most people slow down over the course of the marathon.
        If you have run competitively, you'll know that those who do the opposite—run faster during the second half of the race—are said to have "negative-split" the race.

        Let's create another column in the data, the split fraction, which measures the degree to which each runner negative-splits or positive-splits the race:
        """
    )
    return


@app.cell
def _(data_2):
    data_2['split_frac'] = 1 - 2 * data_2['split_sec'] / data_2['final_sec']
    data_2.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Where this split difference is less than zero, the person negative-split the race by that fraction.
        Let's do a distribution plot of this split fraction (see the following figure):
        """
    )
    return


@app.cell
def _(data_2, plt, sns):
    sns.displot(data_2['split_frac'], kde=False)
    plt.axvline(0, color='k', linestyle='--')
    return


@app.cell
def _(data_2):
    sum(data_2.split_frac < 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Out of nearly 40,000 participants, there were only 250 people who negative-split their marathon.

        Let's see whether there is any correlation between this split fraction and other variables. We'll do this using a `PairGrid`, which draws plots of all these correlations (see the following figure):
        """
    )
    return


@app.cell
def _(data_2, plt, sns):
    _g = sns.PairGrid(data_2, vars=['age', 'split_sec', 'final_sec', 'split_frac'], hue='gender', palette='RdBu_r')
    _g.map(plt.scatter, alpha=0.8)
    _g.add_legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It looks like the split fraction does not correlate particularly with age, but does correlate with the final time: faster runners tend to have closer to even splits on their marathon time. Let's zoom in on the histogram of split fractions separated by gender, shown in the following figure:
        """
    )
    return


@app.cell
def _(data_2, plt, sns):
    sns.kdeplot(data_2.split_frac[data_2.gender == 'M'], label='men', shade=True)
    sns.kdeplot(data_2.split_frac[data_2.gender == 'W'], label='women', shade=True)
    plt.xlabel('split_frac')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The interesting thing here is that there are many more men than women who are running close to an even split!
        It almost looks like a bimodal distribution among the men and women. Let's see if we can suss out what's going on by looking at the distributions as a function of age.

        A nice way to compare distributions is to use a *violin plot*, shown in the following figure:
        """
    )
    return


@app.cell
def _(data_2, sns):
    sns.violinplot(x='gender', y='split_frac', data=data_2, palette=['lightblue', 'lightpink'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's look a little deeper, and compare these violin plots as a function of age (see the following figure). We'll start by creating a new column in the array that specifies the age range that each person is in, by decade:
        """
    )
    return


@app.cell
def _(data_2):
    data_2['age_dec'] = data_2.age.map(lambda age: 10 * (age // 10))
    data_2.head()
    return


@app.cell
def _(data_2, sns):
    men = data_2.gender == 'M'
    women = data_2.gender == 'W'
    with sns.axes_style(style=None):
        sns.violinplot(x='age_dec', y='split_frac', hue='gender', data=data_2, split=True, inner='quartile', palette=['lightblue', 'lightpink'])
    return men, women


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can see where the distributions among men and women differ: the split distributions of men in their 20s to 50s show a pronounced overdensity toward lower splits when compared to women of the same age (or of any age, for that matter).

        Also surprisingly, it appears that the 80-year-old women seem to outperform *everyone* in terms of their split time, although this is likely a small number effect, as there are only a handful of runners in that range:
        """
    )
    return


@app.cell
def _(data_2):
    (data_2.age > 80).sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Back to the men with negative splits: who are these runners? Does this split fraction correlate with finishing quickly? We can plot this very easily. We'll use `regplot`, which will automatically fit a linear regression model to the data (see the following figure):
        """
    )
    return


@app.cell
def _(data_2, plt, sns):
    _g = sns.lmplot(x='final_sec', y='split_frac', col='gender', data=data_2, markers='.', scatter_kws=dict(color='c'))
    _g.map(plt.axhline, y=0.0, color='k', ls=':')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Apparently, among both men and women, the people with fast splits tend to be faster runners who are finishing within ~15,000 seconds, or about 4 hours. People slower than that are much less likely to have a fast second split.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

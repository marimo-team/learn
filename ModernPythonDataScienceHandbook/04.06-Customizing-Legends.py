import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Customizing Plot Legends
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Plot legends give meaning to a visualization, assigning meaning to the various plot elements.
        We previously saw how to create a simple legend; here we'll take a look at customizing the placement and aesthetics of the legend in Matplotlib.

        The simplest legend can be created with the `plt.legend` command, which automatically creates a legend for any labeled plot elements (see the following figure):
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    return (plt,)


@app.cell
def _():
    # '%matplotlib inline\nimport numpy as np' command supported automatically in marimo
    return


@app.cell
def _(np, plt):
    x = np.linspace(0, 10, 1000)
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), '-b', label='Sine')
    ax.plot(x, np.cos(x), '--r', label='Cosine')
    ax.axis('equal')
    _leg = ax.legend()
    return ax, fig, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But there are many ways we might want to customize such a legend.
        For example, we can specify the location and turn on the frame (see the following figure):
        """
    )
    return


@app.cell
def _(ax, fig):
    ax.legend(loc='upper left', frameon=True)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can use the ``ncol`` command to specify the number of columns in the legend, as shown in the following figure:
        """
    )
    return


@app.cell
def _(ax, fig):
    ax.legend(loc='lower center', ncol=2)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And we can use a rounded box (`fancybox`) or add a shadow, change the transparency (alpha value) of the frame, or change the padding around the text (see the following figure):
        """
    )
    return


@app.cell
def _(ax, fig):
    ax.legend(frameon=True, fancybox=True, framealpha=1,
              shadow=True, borderpad=1)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more information on available legend options, see the `plt.legend` docstring.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Choosing Elements for the Legend

        As we have already seen, by default the legend includes all labeled elements from the plot.
        If this is not what is desired, we can fine-tune which elements and labels appear in the legend by using the objects returned by `plot` commands.
        `plt.plot` is able to create multiple lines at once, and returns a list of created line instances.
        Passing any of these to `plt.legend` will tell it which to identify, along with the labels we'd like to specify (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt, x):
    y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
    _lines = plt.plot(x, y)
    plt.legend(_lines[:2], ['first', 'second'], frameon=True)
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        I generally find in practice that it is clearer to use the first method, applying labels to the plot elements you'd like to show on the legend (see the following figure):
        """
    )
    return


@app.cell
def _(plt, x, y):
    plt.plot(x, y[:, 0], label='first')
    plt.plot(x, y[:, 1], label='second')
    plt.plot(x, y[:, 2:])
    plt.legend(frameon=True);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that the legend ignores all elements without a `label` attribute set.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Legend for Size of Points

        Sometimes the legend defaults are not sufficient for the given visualization.
        For example, perhaps you're using the size of points to mark certain features of the data, and want to create a legend reflecting this.
        Here is an example where we'll use the size of points to indicate populations of California cities.
        We'd like a legend that specifies the scale of the sizes of the points, and we'll accomplish this by plotting some labeled data with no entries (see the following figure):
        """
    )
    return


@app.cell
def _():
    # Uncomment to download the data
    # url = ('https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/'
    #        'master/notebooks/data/california_cities.csv')
    # !cd data && curl -O {url}
    return


@app.cell
def _(np, plt):
    import pandas as pd
    cities = pd.read_csv('data/california_cities.csv')

    # Extract the data we're interested in
    lat, lon = cities['latd'], cities['longd']
    population, area = cities['population_total'], cities['area_total_km2']

    # Scatter the points, using size and color but no label
    plt.scatter(lon, lat, label=None,
                c=np.log10(population), cmap='viridis',
                s=area, linewidth=0, alpha=0.5)
    plt.axis('equal')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.colorbar(label='log$_{10}$(population)')
    plt.clim(3, 7)

    # Here we create a legend:
    # we'll plot empty lists with the desired size and label
    for area in [100, 300, 500]:
        plt.scatter([], [], c='k', alpha=0.3, s=area,
                    label=str(area) + ' km$^2$')
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')

    plt.title('California Cities: Area and Population');
    return area, cities, lat, lon, pd, population


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The legend will always reference some object that is on the plot, so if we'd like to display a particular shape we need to plot it.
        In this case, the objects we want (gray circles) are not on the plot, so we fake them by plotting empty lists.
        Recall that the legend only lists plot elements that have a label specified.

        By plotting empty lists, we create labeled plot objects that are picked up by the legend, and now our legend tells us some useful information.
        This strategy can be useful for creating more sophisticated visualizations.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Multiple Legends

        Sometimes when designing a plot you'd like to add multiple legends to the same axes.
        Unfortunately, Matplotlib does not make this easy: via the standard `legend` interface, it is only possible to create a single legend for the entire plot.
        If you try to create a second legend using `plt.legend` or `ax.legend`, it will simply override the first one.
        We can work around this by creating a new legend artist from scratch (`Artist` is the base class Matplotlib uses for visual attributes), and then using the lower-level `ax.add_artist` method to manually add the second artist to the plot (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt):
    fig_1, ax_1 = plt.subplots()
    _lines = []
    styles = ['-', '--', '-.', ':']
    x_1 = np.linspace(0, 10, 1000)
    for i in range(4):
        _lines = _lines + ax_1.plot(x_1, np.sin(x_1 - i * np.pi / 2), styles[i], color='black')
    ax_1.axis('equal')
    ax_1.legend(_lines[:2], ['line A', 'line B'], loc='upper right')
    from matplotlib.legend import Legend
    _leg = Legend(ax_1, _lines[2:], ['line C', 'line D'], loc='lower right')
    ax_1.add_artist(_leg)
    return Legend, ax_1, fig_1, i, styles, x_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is a peek into the low-level artist objects that comprise any Matplotlib plot.
        If you examine the source code of `ax.legend` (recall that you can do this with within the Jupyter notebook using `ax.legend??`) you'll see that the function simply consists of some logic to create a suitable `Legend` artist, which is then saved in the `legend_` attribute and added to the figure when the plot is drawn.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

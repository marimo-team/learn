# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geodatasets==2026.5.1",
#     "geopandas[all]==1.1.3",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
#     "pandas==2.3.0"
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Why matplotlib?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    *An introduction to Matplotlib's place in the Python visualization ecosystem and its design philosophy.*


    Why [Matplotlib](https://matplotlib.org/)? Part of the original Swiss Army Knife of Python data science libraries, Matplotlib is Python's foundational data visualization package–and for good reason: 
    * It forms the backbone of several plotting libraries;
    * it's incredibly well documented; 
    * it supports everything from simple plots to complex and interactive graphs.
    
    In their own words:

    /// admonition | "Matplotlib makes easy things easy and hard things possible"
    ///
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cbook, cm
    from matplotlib.colors import LightSource
    import geopandas
    import geodatasets

    return LightSource, cbook, cm, geodatasets, geopandas, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Some examples
    Here's some examples from the Matplotlib docs and across the web.

    Matplotlib offers a couple interfaces: one functional, one OOP. Let's take a look at the functional interface first:
    """)
    return


@app.cell()
def _(plt):
    # 4 lines of code get us a simple visual :)
    plt.plot([1, 2, 3, 4])
    plt.ylabel("Some numbers")
    plt.title("Basic line plot")

    # get the current `Axes`
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's take a look at the OOP interface:
    """)
    return


@app.cell()
def _(pd, plt):
    climate_data = pd.read_csv(
        "https://assets.datacamp.com/production/repositories/3634/datasets/411add3f8570d5adf891127fd64095020210711b/climate_change.csv",
        parse_dates=True,
        index_col="date",
    )

    # plt.subplots returns a Figure and Axis object. 
    # here, we ask for 2 rows and one column that share an x-axis
    _fig, _ax = plt.subplots(2, 1, sharex=True)

    # Data is plotted by indexing a row  
    _ax[0].plot(
        climate_data.index,
        climate_data['co2']
    )
    _ax[1].plot(
        climate_data.index,
        climate_data['relative_temp']
    )

    # Chart customisations must also be called on each index
    # The axis annotation methods are different to the plt.plot() interface
    _ax[0].set_title("Global rise in C02 and relative temperature")
    _ax[0].set_ylabel("C02 (ppm)")
    _ax[1].set_xlabel("Year")
    _ax[1].set_ylabel("Relative temperature (°C)")

    # Return the Figure
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3d plots
    Everybody thinks 3D plots are cool. Here's an example which is available from the marimo snippets pane:
    """)
    return


@app.cell()
def _(LightSource, cbook, cm, np, plt):
    # Load and format data
    dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
    _z = dem['elevation']
    nrows, ncols = _z.shape
    _x = np.linspace(dem['xmin'], dem['xmax'], ncols)
    _y = np.linspace(dem['ymin'], dem['ymax'], nrows)
    _x, _y = np.meshgrid(_x, _y)

    region = np.s_[5:50, 5:50]
    _x, _y, _z = _x[region], _y[region], _z[region]

    # Set up plot
    _fig, _ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)

    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(_z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = _ax.plot_surface(_x, _y, _z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Maps

    Making maps with matplotlib is easiest using the [`Geopandas`](https://geopandas.org/en/stable/docs/user_guide/mapping.html) library. Geopandas provides spatially enabled dataframes, with a `.plot()` method. Anything that can be used in the `pyplot` interface can generally be used with `GeoDataFrame.plot()`
    """)
    return


@app.cell()
def _(geodatasets, geopandas):
    # Create a GeoDataFrame from example dataset
    chicago = geopandas.read_file(geodatasets.get_path("geoda.chicago_commpop"))

    # Plot a choropleth of population in 2010
    chicago.plot(
        column="POP2010",
        legend=True,
        legend_kwds={"label": "Population in 2010", "orientation": "horizontal"},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding Interactivity
    Naturally, since we're using marimo we may want to add some interactivity to our Matplotlib charts.

    Here's an example with some simple [sliders](https://docs.marimo.io/api/inputs/slider/#marimo.ui.slider):
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_points = mo.ui.slider(
        20,
        100,
        value=50,
        label="Smoothness"
    )
    n_points
    return (n_points,)


@app.cell(hide_code=True)
def _(mo):
    wave_length = mo.ui.slider(
        2,
        10,
        value=5,
        label="Wave length"
    )
    wave_length
    return (wave_length,)


@app.cell
def _(n_points, np, plt, wave_length):
    _x = np.linspace(0, wave_length.value * np.pi, n_points.value)

    plt.plot(_x, np.sin(_x))
    plt.title(f"Sine wave with {n_points.value} points")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Marimo also supports using lasso and box selection tools!

    Use [`mo.ui.matplotlib`](https://docs.marimo.io/api/plotting/#marimo.ui.matplotlib) to make matplotlib plots reactive: select data on the frontend, then use the selection to filter your data in Python.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Design philosophy
    ### Parts of a figure

    At the core of Matplotlib chart architecture are four objects:
    * a `Figure`
    * an `Axes`
    * two or more `Axis` objects
    * `Artists`

    All Axes objects are rendered on the Figure.

    The Axes are the individual sets of indices, starting with at least one X-axis and Y-axis. The Axes methods are the primary interface for configuring most parts of your plot.

    Axis objects provide ticks, tick labels, and scales for the data in the Axes.

    Everything visible on the Figure is an Artist.


    **The canvas and the artist**

    When working with matplotlib, you can use this mental model: the Figure is the canvas, and everything drawn on it is done by artists (lines, text, spines, etc.). When the Figure is rendered, the Artists are drawn to the canvas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="https://matplotlib.org/stable/_images/anatomy.png",
        alt="Parts of a Matplotlib figure",
        width=500,
        height=500,
        caption="An example of a Matplotlib figure, with annotations for the different objects, artists, and code."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coding style
    As we saw earlier, Matplotlib offers two* interfaces for graphing data: `pyplot`, the functional interface, and the object-oriented (OO) style.

    In the Matplotlib documentation, you'll see a mix of both styles, with the OO style generally recommended, especially for complex graphs and reproducible behaviours.

    For quick interactive work, the pyplot interface may be preferred.

    For Marimo, you can use either, but be sure to return the Figure and Axes of the chart you want to see as the last expression in your cell, e.g.

    ```python
    plt.plot([1, 2])
    # plt.gca() gets the current `Axes`
    plt.gca()
    ```

    or

    ```python
    fig, ax = plt.subplots()

    ax.plot([1, 2])
    ax
    ```


    *There's also a third interface, to be used when embedding Matplotlib in GUI applications.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## More
    * Check out the rest of this Marimo Learn course
    * Read the docs for marimo-specific considerations when working with matplotlib: https://docs.marimo.io/guides/working_with_data/plotting/
    * Read the [Matplotlib Quick Start Guide](https://matplotlib.org/stable/users/explain/quick_start.html)
    * Search the marimo snippets list in the developer panel (`ctrl/cmd + j`) for a bunch of Matplotlib examples
    """)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Photometry

    In the previous episode we downloaded photometry data from Pan-STARRS,
    which is available from the same server we have been using to get Gaia data.

    The next step in the analysis is to select candidate stars based on the
    photometry data. The following figure from the paper is a color-magnitude
    diagram showing the stars we previously selected based on proper motion.

    In red is a theoretical isochrone, showing where we expect the stars in GD-1
    to fall based on the metallicity and age of their original globular cluster.

    By selecting stars in the shaded area, we can further distinguish the main
    sequence of GD-1 from mostly younger background stars.

    ## Outline

    1. We will reload the data from the previous episode and make a
       color-magnitude diagram.

    2. We will use an isochrone computed by MIST to specify a polygonal region
       in the color-magnitude diagram and select the stars inside of it.
    """)
    return


@app.cell
def _():
    from os.path import getsize

    import pandas as pd
    from matplotlib import pyplot as plt

    from episode_functions import *
    return getsize, pd, plt


@app.cell
def _(pd):
    filename = 'gd1_data.hdf'
    candidate_df = pd.read_hdf(filename, 'candidate_df')
    return candidate_df, filename


@app.cell
def _(mo):
    mo.md(r"""
    ## Plotting photometry data

    Now that we have photometry data from Pan-STARRS, we can produce a
    color-magnitude diagram to replicate the diagram from the original paper.

    The y-axis shows the apparent magnitude of each source with the g filter.

    The x-axis shows the difference in apparent magnitude between the g and i
    filters, which indicates color.

    With the photometry we downloaded from the PanSTARRS table into `candidate_df`
    we can now recreate this plot.
    """)
    return


@app.cell
def _(candidate_df, plt):
    x_all = candidate_df['g_mean_psf_mag'] - candidate_df['i_mean_psf_mag']
    y_all = candidate_df['g_mean_psf_mag']
    plt.plot(x_all, y_all, 'ko', markersize=0.3, alpha=0.3)

    plt.ylabel('Magnitude (g)')
    plt.xlabel('Color (g-i)')
    return x_all, y_all


@app.cell
def _(mo):
    mo.md(r"""
    We can zoom in on the region of interest by setting the range of x and y
    values displayed with the `xlim` and `ylim` functions. If we put the higher
    value first in the `ylim` call, this will invert the y-axis, putting fainter
    magnitudes at the bottom.
    """)
    return


@app.cell
def _(candidate_df, plt):
    x_zoom = candidate_df['g_mean_psf_mag'] - candidate_df['i_mean_psf_mag']
    y_zoom = candidate_df['g_mean_psf_mag']
    plt.plot(x_zoom, y_zoom, 'ko', markersize=0.3, alpha=0.3)

    plt.ylabel('Magnitude (g)')
    plt.xlabel('Color (g-i)')

    plt.xlim([0, 1.5])
    plt.ylim([22, 14])
    return x_zoom, y_zoom


@app.cell
def _(mo):
    mo.md(r"""
    Our figure does not look exactly like the one in the paper because we
    are working with a smaller region of the sky, so we have fewer stars. But
    the main sequence of GD-1 appears as an overdense region in the lower left.

    We want to be able to make this plot again, with any selection of PanSTARRS
    photometry, so this is a natural time to put it into a function.
    """)
    return


@app.cell
def _(plt):
    def plot_cmd(dataframe):
        """Plot a color magnitude diagram.

        dataframe: DataFrame or Table with photometry data
        """
        y = dataframe['g_mean_psf_mag']
        x = dataframe['g_mean_psf_mag'] - dataframe['i_mean_psf_mag']

        plt.plot(x, y, 'ko', markersize=0.3, alpha=0.3)

        plt.xlim([0, 1.5])
        plt.ylim([22, 14])

        plt.ylabel('Magnitude (g)')
        plt.xlabel('Color (g-i)')
    return (plot_cmd,)


@app.cell
def _(candidate_df, plot_cmd):
    plot_cmd(candidate_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In the next section we will use an isochrone to specify a polygon that
    contains this overdense region.

    ## Isochrone

    Given our understanding of the age, metallicity, and distance to GD-1 we can
    overlay a theoretical isochrone for GD-1 from the MESA Isochrones and Stellar
    Tracks and better identify the main sequence of GD-1.

    **Calculating Isochrone:** We used MESA Isochrones & Stellar Tracks (MIST) to
    compute it, with these parameters:

    - Rotation initial v/v_crit = 0.4
    - Single age, linear scale = 12e9
    - Composition [Fe/H] = -1.35
    - Synthetic Photometry, PanStarrs
    - Extinction av = 0

    ## Making a polygon

    Now we can read in the results which you downloaded as part of the setup instructions.
    """)
    return


@app.cell
def _(pd):
    iso_df = pd.read_hdf('gd1_isochrone.hdf5', 'iso_df')
    iso_df.head()
    return (iso_df,)


@app.cell
def _(candidate_df, iso_df, plot_cmd, plt):
    plot_cmd(candidate_df)
    plt.plot(iso_df['color_g_i'], iso_df['mag_g'])
    return


@app.cell
def _(mo):
    mo.md(r"""
    In the bottom half of the figure, the isochrone passes through the overdense
    region where the stars are likely to belong to GD-1.

    So we will select the part of the isochrone that lies in the overdense region.
    `g_mask` is a Boolean Series that is `True` where `g` is between 18.0 and 21.5.
    """)
    return


@app.cell
def _(iso_df):
    g_all = iso_df['mag_g']

    g_mask = (g_all > 18.0) & (g_all < 21.5)
    g_mask.sum()
    return g_all, g_mask


@app.cell
def _(g_mask, iso_df):
    iso_masked = iso_df[g_mask]
    iso_masked.head()
    return (iso_masked,)


@app.cell
def _(mo):
    mo.md(r"""
    Now, to select the stars in the overdense region, we have to define a polygon
    that includes stars near the isochrone.
    """)
    return


@app.cell
def _(iso_masked):
    g = iso_masked['mag_g']
    left_color = iso_masked['color_g_i'] - 0.06
    right_color = iso_masked['color_g_i'] + 0.12
    return g, left_color, right_color


@app.cell
def _(candidate_df, g, left_color, plot_cmd, plt, right_color):
    plot_cmd(candidate_df)

    plt.plot(left_color, g, label='left color')
    plt.plot(right_color, g, label='right color')

    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Which points are in the polygon?

    Matplotlib provides a `Polygon` object that we can use to check which
    points fall in the polygon we just constructed.

    To make a `Polygon`, we need to assemble `g`, `left_color`, and `right_color`
    into a loop, so the points in `left_color` are connected to the points of
    `right_color` in reverse order.

    We will use a "slice index" to reverse the elements of `right_color`:

    ```python
    reverse_right_color = right_color[::-1]
    ```

    In this example, `start` and `stop` are omitted, which means all elements are
    selected. And `step` is `-1`, which means the elements are in reverse order.
    """)
    return


@app.cell
def _(g, left_color, right_color):
    import numpy as np
    reverse_right_color = right_color[::-1]
    color_loop = np.append(left_color, reverse_right_color)
    color_loop.shape
    return color_loop, np, reverse_right_color


@app.cell
def _(g, np):
    mag_loop = np.append(g, g[::-1])
    mag_loop.shape
    return (mag_loop,)


@app.cell
def _(candidate_df, color_loop, mag_loop, plot_cmd, plt):
    plot_cmd(candidate_df)
    plt.plot(color_loop, mag_loop)
    return


@app.cell
def _(mo):
    mo.md(r"""
    To make a `Polygon`, it will be useful to put `color_loop` and `mag_loop`
    into a `DataFrame`. This is convenient because `Polygon` is expecting an
    Nx2 array, and the `DataFrame` also allows us to save the region we used
    to select stars for reproducibility.
    """)
    return


@app.cell
def _(color_loop, mag_loop, pd):
    loop_df = pd.DataFrame()
    loop_df['color_loop'] = color_loop
    loop_df['mag_loop'] = mag_loop
    loop_df.head()
    return (loop_df,)


@app.cell
def _(loop_df):
    from matplotlib.patches import Polygon

    polygon = Polygon(loop_df)
    polygon
    return Polygon, polygon


@app.cell
def _(mo):
    mo.md(r"""
    The result is a `Polygon` object which has a `contains_points` method.
    This allows us to pass `polygon.contains_points` a list of points and
    for each point it will tell us if the point is contained within the polygon.
    A point is a tuple with two elements, x and y.

    ## Exercise (5 minutes)

    When we encounter a new object, it is good to create a toy example to test
    that it does what we think it does. Define a list of two points (represented
    as two tuples), one that should be inside the polygon and one that should be
    outside the polygon. Call `contains_points` on the polygon we just created,
    passing it the list of points you defined, to verify that the results are
    as expected.
    """)
    return


@app.cell
def _(mo, polygon):
    mo.accordion({
        "Solution": mo.md(r"""
```python
test_points = [(0.4, 20),
               (0.4, 16)]

test_inside_mask = polygon.contains_points(test_points)
test_inside_mask
# array([ True, False])
```

The result is an array of Boolean values, and is as expected.
        """)
    })
    return


@app.cell
def _(polygon):
    test_points = [(0.4, 20),
                   (0.4, 16)]

    test_inside_mask = polygon.contains_points(test_points)
    test_inside_mask
    return test_inside_mask, test_points


@app.cell
def _(mo):
    mo.md(r"""
    ## Save the polygon

    In this episode, we used an isochrone to derive a polygon, which we used to
    select stars based on photometry. So it is important to record the polygon
    as part of the data analysis pipeline.
    """)
    return


@app.cell
def _(filename, loop_df):
    loop_df.to_hdf(filename, key='loop_df')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Selecting based on photometry

    Now we will check how many of the candidate stars are inside the polygon we chose.
    `contains_points` expects a list of (x,y) pairs. We will start by putting color
    and magnitude data from `candidate_df` into a new `DataFrame`.
    """)
    return


@app.cell
def _(candidate_df, pd):
    cmd_df = pd.DataFrame()

    cmd_df['color'] = candidate_df['g_mean_psf_mag'] - candidate_df['i_mean_psf_mag']
    cmd_df['mag'] = candidate_df['g_mean_psf_mag']

    cmd_df.head()
    return (cmd_df,)


@app.cell
def _(cmd_df, polygon):
    inside_mask = polygon.contains_points(cmd_df)
    inside_mask
    return (inside_mask,)


@app.cell
def _(mo):
    mo.md(r"""
    The result is a Boolean array.

    ## Exercise (5 minutes)

    Boolean values are stored as 0s and 1s. `FALSE` = 0 and `TRUE` = 1. Use this
    information to determine the number of stars that fall inside the polygon.
    """)
    return


@app.cell
def _(inside_mask, mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
inside_mask.sum()
# np.int64(486)
```
        """)
    })
    return


@app.cell
def _(inside_mask):
    inside_mask.sum()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now we can use `inside_mask` as a mask to select stars that fall inside the polygon.
    """)
    return


@app.cell
def _(candidate_df, inside_mask):
    winner_df = candidate_df[inside_mask]
    return (winner_df,)


@app.cell
def _(mo):
    mo.md(r"""
    We will make a color-magnitude plot one more time, highlighting the selected
    stars with green markers.
    """)
    return


@app.cell
def _(candidate_df, color_loop, iso_df, mag_loop, plot_cmd, plt, winner_df):
    plot_cmd(candidate_df)
    plt.plot(iso_df['color_g_i'], iso_df['mag_g'])
    plt.plot(color_loop, mag_loop)

    x_win = winner_df['g_mean_psf_mag'] - winner_df['i_mean_psf_mag']
    y_win = winner_df['g_mean_psf_mag']
    plt.plot(x_win, y_win, 'go', markersize=0.5, alpha=0.5)
    return x_win, y_win


@app.cell
def _(mo):
    mo.md(r"""
    The selected stars are, in fact, inside the polygon, which means they have
    photometry data consistent with GD-1.

    Finally, we can plot the coordinates of the selected stars.
    """)
    return


@app.cell
def _(plt, winner_df):
    fig_winner = plt.figure(figsize=(10, 2.5))

    x_phi = winner_df['phi1']
    y_phi = winner_df['phi2']
    plt.plot(x_phi, y_phi, 'ko', markersize=0.7, alpha=0.9)

    plt.xlabel(r'$\phi_1$ [deg]')
    plt.ylabel(r'$\phi_2$ [deg]')
    plt.title('Proper motion + photometry selection', fontsize='medium')

    plt.axis('equal')
    return fig_winner, x_phi, y_phi


@app.cell
def _(mo):
    mo.md(r"""
    In the next episode we are going to make this plot several more times, so it
    makes sense to make a function. As we have done with previous functions we can
    copy and paste what we just wrote, replacing the specific variable `winner_df`
    with the more generic `df`.
    """)
    return


@app.cell
def _(plt):
    def plot_cmd_selection(df):
        x = df['phi1']
        y = df['phi2']

        plt.plot(x, y, 'ko', markersize=0.7, alpha=0.9)

        plt.xlabel(r'$\phi_1$ [deg]')
        plt.ylabel(r'$\phi_2$ [deg]')
        plt.title('Proper motion + photometry selection', fontsize='medium')

        plt.axis('equal')
    return (plot_cmd_selection,)


@app.cell
def _(plot_cmd_selection, plt, winner_df):
    fig_sel = plt.figure(figsize=(10, 2.5))
    plot_cmd_selection(winner_df)
    return (fig_sel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Write the data

    Finally, we will write the selected stars to a file.
    """)
    return


@app.cell
def _(filename, winner_df):
    winner_df.to_hdf(filename, key='winner_df')
    return


@app.cell
def _(filename, getsize):
    # 1 MB = 1024 * 1024 bytes
    MB = 1024 * 1024
    getsize(filename) / MB
    return (MB,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    In this episode, we used photometry data from Pan-STARRS to draw a
    color-magnitude diagram. We used an isochrone to define a polygon and select
    stars we think are likely to be in GD-1. Plotting the results, we have a
    clearer picture of GD-1, similar to Figure 1 in the original paper.

    **Key points:**

    - Matplotlib provides operations for working with points, polygons, and other
      geometric entities, so it is not just for making figures.
    - Use Matplotlib options to control the size and aspect ratio of figures to make
      them easier to interpret.
    - Record every element of the data analysis pipeline that would be needed to
      replicate the results.
    """)
    return

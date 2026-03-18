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
    # Plotting and Pandas

    In the previous episode, we wrote a query to select stars from the
    region of the sky where we expect GD-1 to be, and saved the results in
    FITS and HDF5 files.

    Now we will read that data back in and implement the next step in the
    analysis, identifying stars with the proper motion we expect for GD-1.

    We will:
    1. Select stars near the centerline of GD-1
    2. Plot their proper motion to identify a cluster at a non-zero value
    3. Select stars whose proper motion is in that cluster region
    """)
    return


@app.cell
def _():
    import astropy.units as u
    import matplotlib.pyplot as plt
    import pandas as pd
    return (pd, plt, u)


@app.cell
def _(pd):
    results_df = pd.read_hdf('gd1_data.hdf', 'results_df')
    return (results_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exploring Data

    One benefit of using Pandas is that it provides functions for
    exploring the data and checking for problems. `describe` computes
    summary statistics for each column.
    """)
    return


@app.cell
def _(results_df):
    results_df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (10 minutes)

    Review the summary statistics in this table.

    - Do the values make sense based on what you know about the context?
    - Do you see any values that seem problematic, or evidence of other data issues?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    The most noticeable issue is that some of the parallax values are negative,
    which seems non-physical.

    Negative parallaxes in the Gaia database can arise from source confusion
    (high negative values) and the parallax zero point with systematic errors
    (low negative values).

    Fortunately, we do not use the parallax measurements in the analysis —
    one of the reasons we used a constant distance for reflex correction.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Plot Proper Motion

    Now we are ready to replicate one of the panels in Figure 1 of the
    Price-Whelan and Bonaca paper, which shows components of proper motion
    as a scatter plot.

    Due to the nature of tidal streams, we expect:
    - Proper motion in the phi2 direction to be near 0
    - Proper motion in phi1 to form a cluster at a non-zero value

    We start with a full scatter plot and then zoom in on the region near
    the origin.
    """)
    return


@app.cell
def _(plt, results_df):
    x_all = results_df['pm_phi1']
    y_all = results_df['pm_phi2']
    plt.plot(x_all, y_all, 'ko', markersize=0.1, alpha=0.1)

    plt.xlabel('Proper motion phi1 (mas/yr GD1 frame)')
    plt.ylabel('Proper motion phi2 (mas/yr GD1 frame)')
    plt.gca()
    return (x_all, y_all)


@app.cell
def _(plt, results_df):
    # Zoom in on the region near the origin following the paper's example
    x_zoom = results_df['pm_phi1']
    y_zoom = results_df['pm_phi2']
    plt.plot(x_zoom, y_zoom, 'ko', markersize=0.1, alpha=0.1)

    plt.xlabel('Proper motion phi1 (mas/yr GD1 frame)')
    plt.ylabel('Proper motion phi2 (mas/yr GD1 frame)')

    plt.xlim(-12, 8)
    plt.ylim(-10, 10)
    plt.gca()
    return (x_zoom, y_zoom)


@app.cell
def _(mo):
    mo.md(r"""
    There is a hint of an overdense region near (-7.5, 0). To see the cluster
    more clearly, we need a sample with a higher proportion of GD-1 stars.
    We will do that by selecting stars close to the centerline (phi2 near 0).

    ## Selecting the Centerline

    Many stars in GD-1 are less than 1 degree from the line phi2=0. Stars
    near this line have the highest probability of being in GD-1.

    We use a "Boolean mask" to select them.
    """)
    return


@app.cell
def _(results_df, u):
    phi2 = results_df['phi2']

    phi2_min = -1.0 * u.degree
    phi2_max = 1.0 * u.degree

    # Use & for elementwise logical AND (Python's 'and' doesn't work with Pandas)
    mask = (phi2 > phi2_min) & (phi2 < phi2_max)
    mask.head()
    return (mask, phi2, phi2_max, phi2_min)


@app.cell
def _(mask):
    # The sum of a Boolean Series is the number of True values
    mask.sum()
    return


@app.cell
def _(mask, results_df):
    centerline_df = results_df[mask]
    len(centerline_df)
    return (centerline_df,)


@app.cell
def _(centerline_df, results_df):
    # What fraction of the rows were selected?
    len(centerline_df) / len(results_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    About 25,000 stars (18% of the total) are near the centerline.

    ## Plotting Proper Motion

    We will write a reusable function to plot proper motion for any DataFrame.
    """)
    return


@app.cell
def _(plt):
    def plot_proper_motion(df):
        """Plot proper motion.

        df: DataFrame with `pm_phi1` and `pm_phi2`
        """
        x = df['pm_phi1']
        y = df['pm_phi2']
        plt.plot(x, y, 'ko', markersize=0.3, alpha=0.3)

        plt.xlabel('Proper motion phi1 (mas/yr)')
        plt.ylabel('Proper motion phi2 (mas/yr)')

        plt.xlim(-12, 8)
        plt.ylim(-10, 10)
    return (plot_proper_motion,)


@app.cell
def _(centerline_df, plot_proper_motion, plt):
    plot_proper_motion(centerline_df)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now we can see more clearly that there is a cluster near (-7.5, 0).
    These are the stars that are likely to be in GD-1.

    ## Filtering Based on Proper Motion

    We will select stars in the overdense region of proper motion using
    bounds we chose by eye.
    """)
    return


@app.cell
def _():
    # Proper motion bounds chosen by eye to encompass the overdense cluster
    pm1_min = -8.9
    pm1_max = -6.9
    pm2_min = -2.2
    pm2_max = 1.0
    return (pm1_max, pm1_min, pm2_max, pm2_min)


@app.cell
def _(pm1_max, pm1_min, pm2_max, pm2_min):
    def make_rectangle(x1, x2, y1, y2):
        """Return the corners of a rectangle."""
        xs = [x1, x1, x2, x2, x1]
        ys = [y1, y2, y2, y1, y1]
        return xs, ys

    pm1_rect, pm2_rect = make_rectangle(
        pm1_min, pm1_max, pm2_min, pm2_max)
    return (make_rectangle, pm1_rect, pm2_rect)


@app.cell
def _(centerline_df, plot_proper_motion, plt, pm1_rect, pm2_rect):
    # Plot proper motion with the selection rectangle overlaid
    plot_proper_motion(centerline_df)
    plt.plot(pm1_rect, pm2_rect, '-')
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now that we have identified the bounds of the cluster, we will use it
    to select rows from `results_df`.

    We write a helper function that uses Pandas operators to make a mask
    that selects rows where a series falls between `low` and `high`.
    """)
    return


@app.cell
def _():
    def between(series, low, high):
        """Check whether values are between `low` and `high`."""
        return (series > low) & (series < high)
    return (between,)


@app.cell
def _(between, pm1_max, pm1_min, pm2_max, pm2_min, results_df):
    pm1 = results_df['pm_phi1']
    pm2 = results_df['pm_phi2']

    pm_mask = (between(pm1, pm1_min, pm1_max) &
               between(pm2, pm2_min, pm2_max))
    pm_mask.sum()
    return (pm1, pm2, pm_mask)


@app.cell
def _(pm_mask, results_df):
    selected_df = results_df[pm_mask]
    len(selected_df)
    return (selected_df,)


@app.cell
def _(mo):
    mo.md(r"""
    These are the stars we think are likely to be in GD-1. We can inspect
    them by plotting their coordinates (not their proper motion).
    """)
    return


@app.cell
def _(plt, selected_df):
    x_sel = selected_df['phi1']
    y_sel = selected_df['phi2']
    plt.plot(x_sel, y_sel, 'ko', markersize=1, alpha=1)

    plt.xlabel('phi1 (degree GD1)')
    plt.ylabel('phi2 (degree GD1)')
    plt.gca()
    return (x_sel, y_sel)


@app.cell
def _(mo):
    mo.md(r"""
    Now that is starting to look like a tidal stream!

    We will write a reusable function to create this plot and add a title
    and equal axes for better visualization.
    """)
    return


@app.cell
def _(plt):
    def plot_pm_selection(df):
        """Plot in GD-1 spatial coordinates the location of the stars
        selected by proper motion.
        """
        x = df['phi1']
        y = df['phi2']

        plt.plot(x, y, 'ko', markersize=0.3, alpha=0.3)

        plt.xlabel('phi1 [deg]')
        plt.ylabel('phi2 [deg]')
        plt.title('Proper motion selection', fontsize='medium')

        plt.axis('equal')
    return (plot_pm_selection,)


@app.cell
def _(plot_pm_selection, plt, selected_df):
    plot_pm_selection(selected_df)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Saving the DataFrames

    At this point we have run a successful query and cleaned up the results.
    We save both DataFrames to the HDF5 file we started in the previous episode.

    We add `selected_df` as a new Dataset (without `mode='w'` to avoid
    overwriting the existing `results_df` Dataset).
    """)
    return


@app.cell
def _(selected_df):
    filename = 'gd1_data.hdf'
    selected_df.to_hdf(filename, key='selected_df')
    return (filename,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    We are going to need `centerline_df` later as well. Write a line of
    code to add it as a second Dataset in the HDF5 file.

    Hint: Since the file already exists, you should *not* use `mode='w'`.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    ```python
    centerline_df.to_hdf(filename, key='centerline_df')
    ```
        """)
    })
    return


@app.cell
def _(centerline_df, filename):
    centerline_df.to_hdf(filename, key='centerline_df')
    return


@app.cell
def _(filename, pd):
    # Verify all datasets are in the file
    with pd.HDFStore(filename) as hdf:
        print(hdf.keys())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    In this episode, we re-loaded the transformed Gaia data and prototyped
    the selection process from the Price-Whelan and Bonaca paper:

    - Selected stars near the centerline of GD-1 and made a scatter plot
      of their proper motion
    - Identified a region of proper motion containing stars likely in GD-1
    - Used a Boolean `Series` as a mask to select stars with proper motion
      in that region
    - Saved both DataFrames to an HDF5 file for use in future episodes

    Key points:
    - A workflow is often prototyped on a small set of data to identify
      filters that limit the dataset to exactly what you want
    - To store data from a Pandas `DataFrame`, HDF5 is a good option that
      can contain multiple Datasets
    """)
    return


if __name__ == "__main__":
    app.run()

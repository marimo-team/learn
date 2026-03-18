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
    # Plotting and Tabular Data

    In the previous episode, we wrote a query to select stars from the
    region of the sky where we expect GD-1 to be, and saved the results in
    a FITS file.

    Now we will read that data back in and implement the next step in the
    analysis, identifying stars with the proper motion we expect for GD-1.

    We will:
    1. Read back the results from the previous lesson, saved in a FITS file
    2. Transform the coordinates and proper motion data from ICRS to the GD-1 frame
    3. Put the results into a Pandas `DataFrame`
    """)
    return


@app.cell
def _():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import matplotlib.pyplot as plt
    import pandas as pd
    from gd1 import GD1Koposov10
    from reflex import reflex_correct
    return (
        GD1Koposov10,
        SkyCoord,
        Table,
        pd,
        plt,
        reflex_correct,
        u,
    )


@app.cell
def _(GD1Koposov10, Table):
    polygon_results = Table.read('gd1_results.fits')
    gd1_frame = GD1Koposov10()
    return (gd1_frame, polygon_results)


@app.cell
def _(mo):
    mo.md(r"""
    ## Selecting Rows and Columns

    We can use `info` to check the contents of the table.
    """)
    return


@app.cell
def _(polygon_results):
    polygon_results.info()
    return


@app.cell
def _(polygon_results):
    # Get column names
    polygon_results.colnames
    return


@app.cell
def _(polygon_results):
    # Select an individual column
    polygon_results['ra']
    return


@app.cell
def _(polygon_results):
    # Select the first row
    polygon_results[0]
    return


@app.cell
def _(polygon_results):
    # Select a column and then an element
    polygon_results['ra'][0]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scatter Plot

    To see what the results look like, we will use a scatter plot with
    Matplotlib. We plot right ascension vs. declination in ICRS coordinates.
    """)
    return


@app.cell
def _(plt, polygon_results):
    x = polygon_results['ra']
    y = polygon_results['dec']
    plt.plot(x, y, 'ko')

    plt.xlabel('ra (degree ICRS)')
    plt.ylabel('dec (degree ICRS)')
    plt.gca()
    return (x, y)


@app.cell
def _(mo):
    mo.md(r"""
    This scatter plot is "overplotted" — there are so many overlapping
    points that we cannot distinguish between high and low density areas.
    We can fix this by controlling the size and transparency of the points.

    ## Exercise (5 minutes)

    In the call to `plt.plot`, use the keyword argument `markersize` to
    make the markers smaller. Then add the keyword argument `alpha` to
    make the markers partly transparent. Adjust these arguments until you
    think the figure shows the data most clearly.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    ```python
    x = polygon_results['ra']
    y = polygon_results['dec']
    plt.plot(x, y, 'ko', markersize=0.1, alpha=0.1)

    plt.xlabel('ra (degree ICRS)')
    plt.ylabel('dec (degree ICRS)')
    ```

    With `markersize=0.1` and `alpha=0.1`, the stripes caused by Gaia's
    scanning law become visible.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Transform Back to GD-1 Frame

    We will transform the ICRS coordinates back to the GD-1 frame so the
    axes are aligned with the orbit of GD-1. This makes it easier to:

    - Identify stars near the centerline of the stream (where phi2 is close to 0)
    - Identify stars with proper motion along phi1 that are likely in GD-1

    We create a `SkyCoord` object with position, proper motions, a constant
    distance estimate, and a placeholder radial velocity.
    """)
    return


@app.cell
def _(SkyCoord, polygon_results, u):
    # Use a constant distance estimate of 8 kpc for GD-1
    # (individual parallax measurements are unreliable at this distance)
    # Radial velocity is set to 0 as a placeholder for the reflex correction
    distance = 8 * u.kpc
    radial_velocity = 0 * u.km/u.s

    skycoord = SkyCoord(ra=polygon_results['ra'],
                        dec=polygon_results['dec'],
                        pm_ra_cosdec=polygon_results['pmra'],
                        pm_dec=polygon_results['pmdec'],
                        distance=distance,
                        radial_velocity=radial_velocity)
    return (distance, radial_velocity, skycoord)


@app.cell
def _(gd1_frame, skycoord):
    transformed = skycoord.transform_to(gd1_frame)
    return (transformed,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Reflex Correction

    The next step is to correct the proper motion measurements for the
    effect of the motion of our solar system around the Galactic center.
    We use `reflex_correct` from Gala for this.
    """)
    return


@app.cell
def _(reflex_correct, transformed):
    skycoord_gd1 = reflex_correct(transformed)
    return (skycoord_gd1,)


@app.cell
def _(mo):
    mo.md(r"""
    Now we can plot phi1 vs phi2 — the coordinates in the GD-1 frame.
    """)
    return


@app.cell
def _(plt, skycoord_gd1):
    x_gd1 = skycoord_gd1.phi1
    y_gd1 = skycoord_gd1.phi2
    plt.plot(x_gd1, y_gd1, 'ko', markersize=0.1, alpha=0.1)

    plt.xlabel('phi1 (degree GD1)')
    plt.ylabel('phi2 (degree GD1)')
    plt.gca()
    return (x_gd1, y_gd1)


@app.cell
def _(mo):
    mo.md(r"""
    We started with a rectangle in the GD-1 frame, which became a
    non-rectangular region in ICRS, and is now a rectangle again after
    transforming back.

    ## Pandas DataFrame

    Now we extract the GD-1 frame columns from `skycoord_gd1` and add
    them to the Astropy `Table` `polygon_results`.
    """)
    return


@app.cell
def _(polygon_results, skycoord_gd1):
    polygon_results['phi1'] = skycoord_gd1.phi1
    polygon_results['phi2'] = skycoord_gd1.phi2
    polygon_results.info()
    return


@app.cell
def _(polygon_results, skycoord_gd1):
    polygon_results['pm_phi1'] = skycoord_gd1.pm_phi1_cosphi2
    polygon_results['pm_phi2'] = skycoord_gd1.pm_phi2
    polygon_results.info()
    return


@app.cell
def _(mo):
    mo.md(r"""
    It is straightforward to convert an Astropy `Table` to a Pandas `DataFrame`.
    """)
    return


@app.cell
def _(pd, polygon_results):
    results_df = polygon_results.to_pandas()
    results_df.shape
    return (results_df,)


@app.cell
def _(results_df):
    results_df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The `make_dataframe` Function

    We consolidate all the transformation steps into a single reusable
    function that takes an Astropy `Table` from a Gaia query and returns
    a Pandas `DataFrame` with GD-1 coordinates and proper motions added.
    """)
    return


@app.cell
def _(GD1Koposov10, SkyCoord, reflex_correct, u):
    def make_dataframe(table):
        """Transform coordinates from ICRS to GD-1 frame.

        table: Astropy Table

        returns: Pandas DataFrame
        """
        # Create a SkyCoord object with the coordinates and proper motions
        # in the input table
        skycoord = SkyCoord(
                   ra=table['ra'],
                   dec=table['dec'],
                   pm_ra_cosdec=table['pmra'],
                   pm_dec=table['pmdec'],
                   distance=8*u.kpc,
                   radial_velocity=0*u.km/u.s)

        # Define the GD-1 reference frame
        gd1_frame = GD1Koposov10()

        # Transform input coordinates to the GD-1 reference frame
        transformed = skycoord.transform_to(gd1_frame)

        # Correct GD-1 coordinates for solar system motion around galactic center
        skycoord_gd1 = reflex_correct(transformed)

        # Add GD-1 reference frame columns for coordinates and proper motions
        table['phi1'] = skycoord_gd1.phi1
        table['phi2'] = skycoord_gd1.phi2
        table['pm_phi1'] = skycoord_gd1.pm_phi1_cosphi2
        table['pm_phi2'] = skycoord_gd1.pm_phi2

        # Create DataFrame
        df = table.to_pandas()

        return df
    return (make_dataframe,)


@app.cell
def _(Table, make_dataframe):
    # Re-read the original table and apply the full transformation
    polygon_results_fresh = Table.read('gd1_results.fits')
    results_df_final = make_dataframe(polygon_results_fresh)
    results_df_final.head()
    return (polygon_results_fresh, results_df_final)


@app.cell
def _(mo):
    mo.md(r"""
    ## Saving the DataFrame

    At this point we have run a successful query and combined the results
    into a single `DataFrame`. This is a good time to save the data.

    One of the best options is HDF5 (Hierarchical Data Format version 5),
    which is a binary format that is small, fast, cross-language, and can
    contain multiple datasets with metadata.
    """)
    return


@app.cell
def _(results_df_final):
    filename = 'gd1_data.hdf'
    results_df_final.to_hdf(filename, key='results_df', mode='w')
    return (filename,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    In this episode, we:
    - Re-loaded the Gaia data saved from a previous query
    - Transformed coordinates and proper motion from ICRS to GD-1 frame
    - Applied reflex correction for the solar system's motion
    - Stored the results in a Pandas `DataFrame`
    - Saved the results to an HDF5 file

    Key points:
    - When you make a scatter plot, adjust marker size and transparency to avoid overplotting
    - For simple scatter plots in Matplotlib, `plot` is faster than `scatter`
    - An Astropy `Table` and a Pandas `DataFrame` are similar; choose based on your needs
    - To store data from a Pandas `DataFrame`, HDF5 is a good option
    """)
    return


if __name__ == "__main__":
    app.run()

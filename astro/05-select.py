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
    # Transform and Select

    In the previous episode, we identified stars with the proper motion we
    expect for GD-1.

    Now we will do the same selection in an ADQL query, which will make it
    possible to work with a larger region of the sky and still download
    less data.

    ## Outline

    1. Using data from the previous episode, we will identify the values of
       proper motion for stars likely to be in GD-1.

    2. Then we will compose an ADQL query that selects stars based on proper
       motion, so we can download only the data we need.

    That will make it possible to search a bigger region of the sky in a
    single query.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Starting from this episode

    If you are starting a new notebook for this episode, you will need to run
    the following setup code. The imports and data loading below assume your
    notebook is being run in the `student_download` directory.
    """)
    return


@app.cell
def _():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.gaia import Gaia
    import matplotlib.pyplot as plt
    import pandas as pd

    from episode_functions import *
    from gd1 import GD1Koposov10
    from reflex import reflex_correct
    return (
        Gaia,
        GD1Koposov10,
        SkyCoord,
        pd,
        plt,
        reflex_correct,
        u,
    )


@app.cell
def _(GD1Koposov10, pd):
    filename = 'gd1_data.hdf'
    centerline_df = pd.read_hdf(filename, 'centerline_df')
    selected_df = pd.read_hdf(filename, 'selected_df')
    return centerline_df, filename, selected_df


@app.cell
def _(GD1Koposov10):
    pm1_min = -8.9
    pm1_max = -6.9
    pm2_min = -2.2
    pm2_max = 1.0

    pm1_rect, pm2_rect = make_rectangle(
        pm1_min, pm1_max, pm2_min, pm2_max)
    gd1_frame = GD1Koposov10()
    return gd1_frame, pm1_max, pm1_min, pm1_rect, pm2_max, pm2_min, pm2_rect


@app.cell
def _(mo):
    mo.md(r"""
    ## Selection by proper motion

    Let us review how we got to this point.

    1. We made an ADQL query to the Gaia server to get data for stars in
       the vicinity of a small part of GD-1.

    2. We transformed the coordinates to the GD-1 frame (`GD1Koposov10`) so we
       could select stars along the centerline of GD-1.

    3. We plotted the proper motion of stars along the centerline of GD-1
       to identify the bounds of an anomalous overdense region associated
       with the proper motion of stars in GD-1.

    4. We made a mask that selects stars whose proper motion is in this
       overdense region and which are therefore likely to be part of the GD-1 stream.

    At this point we have downloaded data for a relatively large number of
    stars (more than 100,000) and selected a relatively small number
    (around 1000).

    It would be more efficient to use ADQL to select only the stars we
    need. That would also make it possible to download data covering a
    larger region of the sky.

    However, the selection we did was based on proper motion in the
    GD-1 frame. In order to do the same selection on the Gaia catalog in ADQL,
    we have to work with proper motions in the ICRS frame as this is the
    frame that the Gaia catalog uses.

    First, we will verify that our proper motion selection was correct,
    starting with the `plot_proper_motion` function that we defined in episode 3.
    The following figure shows:

    - Proper motion for the stars we selected along the center line of GD-1,
    - The rectangle we selected, and
    - The stars inside the rectangle highlighted in green.
    """)
    return


@app.cell
def _(centerline_df, plt, pm1_rect, pm2_rect, selected_df):
    plot_proper_motion(centerline_df)

    plt.plot(pm1_rect, pm2_rect)

    x = selected_df['pm_phi1']
    y = selected_df['pm_phi2']
    plt.plot(x, y, 'gx', markersize=0.3, alpha=0.3)
    return x, y


@app.cell
def _(mo):
    mo.md(r"""
    Now we will make the same plot using proper motions in the ICRS frame,
    which are stored in columns named `pmra` and `pmdec`.
    """)
    return


@app.cell
def _(centerline_df, plt, selected_df):
    x_icrs = centerline_df['pmra']
    y_icrs = centerline_df['pmdec']
    plt.plot(x_icrs, y_icrs, 'ko', markersize=0.3, alpha=0.3)

    x_sel = selected_df['pmra']
    y_sel = selected_df['pmdec']
    plt.plot(x_sel, y_sel, 'gx', markersize=1, alpha=0.3)

    plt.xlabel('Proper motion ra (ICRS frame)')
    plt.ylabel('Proper motion dec (ICRS frame)')

    plt.xlim([-10, 5])
    plt.ylim([-20, 5])
    return x_icrs, x_sel, y_icrs, y_sel


@app.cell
def _(mo):
    mo.md(r"""
    The proper motions of the selected stars are more spread out in this
    frame, which is why it was preferable to do the selection in the GD-1 frame.

    In the following exercise, we will identify a rectangle that encompasses
    the majority of the stars we identified as having proper motion consistent
    with that of GD-1 without including too many other stars.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    Looking at the proper motion of the stars we identified along the centerline
    of GD-1, in the ICRS reference frame define a rectangle (`pmra_min`,
    `pmra_max`, `pmdec_min`, and `pmdec_max`) that encompass the proper motion
    of the majority of the stars near the centerline of GD-1 without including
    too much contamination from other stars.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
pmra_min = -6.70
pmra_max = -3
pmdec_min = -14.31
pmdec_max = -11.2
```
        """)
    })
    return


@app.cell
def _():
    pmra_min = -6.70
    pmra_max = -3
    pmdec_min = -14.31
    pmdec_max = -11.2
    return pmra_max, pmra_min, pmdec_max, pmdec_min


@app.cell
def _(mo):
    mo.md(r"""
    ## Assembling the query

    In episode 2 we used the following query to select stars in a polygonal region
    around a small part of GD-1 with a few filters on color and distance (parallax).

    In this episode we will make two changes:

    1. We will select stars with coordinates in a larger region to include more of GD-1.

    2. We will add another clause to select stars whose proper motion is in
       the range we just defined in the previous exercise.

    The fact that we remove most contaminating stars with the proper motion filter
    is what allows us to expand our query to include most of GD-1 without returning
    too many results.

    As we did in episode 2, we will define the physical region we want to select in
    the GD-1 frame and transform it to the ICRS frame to query the Gaia catalog.

    Here are the coordinates of the larger rectangle in the GD-1 frame.
    """)
    return


@app.cell
def _(u):
    candidate_coord_query_base = """SELECT
    {columns}
    FROM gaiadr2.gaia_source
    WHERE parallax < 1
      AND bp_rp BETWEEN -0.75 AND 2
      AND 1 = CONTAINS(POINT(ra, dec),
                       POLYGON({sky_point_list}))
    """
    return (candidate_coord_query_base,)


@app.cell
def _(u):
    phi1_min = -70 * u.degree
    phi1_max = -20 * u.degree
    phi2_min = -5 * u.degree
    phi2_max = 5 * u.degree
    return phi1_max, phi1_min, phi2_max, phi2_min


@app.cell
def _(phi1_max, phi1_min, phi2_max, phi2_min):
    phi1_rect, phi2_rect = make_rectangle(
        phi1_min, phi1_max, phi2_min, phi2_max)
    return phi1_rect, phi2_rect


@app.cell
def _(SkyCoord, gd1_frame, phi1_rect, phi2_rect):
    corners = SkyCoord(phi1=phi1_rect,
                       phi2=phi2_rect,
                       frame=gd1_frame)

    corners_icrs = corners.transform_to('icrs')
    return corners, corners_icrs


@app.cell
def _(corners_icrs):
    sky_point_list = skycoord_to_string(corners_icrs)
    sky_point_list
    return (sky_point_list,)


@app.cell
def _():
    columns = 'source_id, ra, dec, pmra, pmdec'
    return (columns,)


@app.cell
def _(mo):
    mo.md(r"""
    Now we have everything we need to assemble the query, but
    **DO NOT try to run this query**.
    Because it selects a larger region, there are too many stars to handle
    in a single query. Until we select by proper motion, that is.
    """)
    return


@app.cell
def _(candidate_coord_query_base, columns, sky_point_list):
    candidate_coord_query = candidate_coord_query_base.format(
        columns=columns,
        sky_point_list=sky_point_list)
    print(candidate_coord_query)
    return (candidate_coord_query,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Selecting proper motion

    Now we are ready to add a `WHERE` clause to select stars whose proper
    motion falls in the range we defined in the last exercise.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (10 minutes)

    Define `candidate_coord_pm_query_base`, starting with `candidate_coord_query_base`
    and adding two new `BETWEEN` clauses to select stars whose coordinates of proper
    motion, `pmra` and `pmdec`, fall within the region defined by `pmra_min`,
    `pmra_max`, `pmdec_min`, and `pmdec_max`.
    In the next exercise we will use the format statement to fill in the values
    we defined above.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
candidate_coord_pm_query_base = """SELECT
{columns}
FROM gaiadr2.gaia_source
WHERE parallax < 1
  AND bp_rp BETWEEN -0.75 AND 2
  AND 1 = CONTAINS(POINT(ra, dec),
                   POLYGON({sky_point_list}))
  AND pmra BETWEEN {pmra_min} AND  {pmra_max}
  AND pmdec BETWEEN {pmdec_min} AND {pmdec_max}
"""
```
        """)
    })
    return


@app.cell
def _():
    candidate_coord_pm_query_base = """SELECT
{columns}
FROM gaiadr2.gaia_source
WHERE parallax < 1
  AND bp_rp BETWEEN -0.75 AND 2
  AND 1 = CONTAINS(POINT(ra, dec),
                   POLYGON({sky_point_list}))
  AND pmra BETWEEN {pmra_min} AND  {pmra_max}
  AND pmdec BETWEEN {pmdec_min} AND {pmdec_max}
"""
    return (candidate_coord_pm_query_base,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    Use `format` to format `candidate_coord_pm_query_base` and define
    `candidate_coord_pm_query`, filling in the values of `columns`,
    `sky_point_list`, and `pmra_min`, `pmra_max`, `pmdec_min`, `pmdec_max`.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
candidate_coord_pm_query = candidate_coord_pm_query_base.format(
    columns=columns,
    sky_point_list=sky_point_list,
    pmra_min=pmra_min,
    pmra_max=pmra_max,
    pmdec_min=pmdec_min,
    pmdec_max=pmdec_max)
print(candidate_coord_pm_query)
```
        """)
    })
    return


@app.cell
def _(
    candidate_coord_pm_query_base,
    columns,
    pmra_max,
    pmra_min,
    pmdec_max,
    pmdec_min,
    sky_point_list,
):
    candidate_coord_pm_query = candidate_coord_pm_query_base.format(
        columns=columns,
        sky_point_list=sky_point_list,
        pmra_min=pmra_min,
        pmra_max=pmra_max,
        pmdec_min=pmdec_min,
        pmdec_max=pmdec_max)
    print(candidate_coord_pm_query)
    return (candidate_coord_pm_query,)


@app.cell
def _(mo):
    mo.md(r"""
    Now we can run the query. This launches an asynchronous job on the Gaia server.
    """)
    return


@app.cell
def _(Gaia, candidate_coord_pm_query):
    candidate_coord_pm_job = Gaia.launch_job_async(candidate_coord_pm_query)
    print(candidate_coord_pm_job)
    return (candidate_coord_pm_job,)


@app.cell
def _(candidate_coord_pm_job):
    candidate_gaia_table = candidate_coord_pm_job.get_results()
    len(candidate_gaia_table)
    return (candidate_gaia_table,)


@app.cell
def _(mo):
    mo.md(r"""
    We call the results `candidate_gaia_table` because it contains information
    from the Gaia table for stars that are good candidates for GD-1.

    `sky_point_list`, `pmra_min`, `pmra_max`, `pmdec_min`, and `pmdec_max` are
    a set of selection criteria that we derived from data downloaded from the
    Gaia Database. To make sure we can repeat our analysis at a later date we
    should save this information to a file.

    We will save them in a Pandas `Series` stored in an HDF5 file.

    **Note on BETWEEN vs POLYGON:** ADQL intends the `POLYGON` function to only be
    used on coordinates and not on proper motion. To enforce this, it will produce
    an error when a negative value is passed into the first argument. This is why
    we used `BETWEEN` for proper motion.
    """)
    return


@app.cell
def _(pd, pmra_max, pmra_min, pmdec_max, pmdec_min, sky_point_list):
    d = dict(sky_point_list=sky_point_list,
             pmra_min=pmra_min,
             pmra_max=pmra_max,
             pmdec_min=pmdec_min,
             pmdec_max=pmdec_max)
    d
    return (d,)


@app.cell
def _(d, pd):
    point_series = pd.Series(d)
    point_series
    return (point_series,)


@app.cell
def _(filename, point_series):
    point_series.to_hdf(filename, key='point_series')
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Performance Warning:** You may see a PerformanceWarning about PyTables
    pickling object types. This is because we are mixing strings and floats in a
    single Series. The amount of data is small enough that this warning can be
    safely ignored.

    ## Plotting one more time

    Now we can examine the results.
    """)
    return


@app.cell
def _(candidate_gaia_table, plt):
    x_res = candidate_gaia_table['ra']
    y_res = candidate_gaia_table['dec']
    plt.plot(x_res, y_res, 'ko', markersize=0.3, alpha=0.3)

    plt.xlabel('ra (degree ICRS)')
    plt.ylabel('dec (degree ICRS)')
    return x_res, y_res


@app.cell
def _(mo):
    mo.md(r"""
    This plot shows why it was useful to transform these coordinates to the GD-1
    frame. In ICRS, it is more difficult to identify the stars near the centerline
    of GD-1.

    We can use our `make_dataframe` function from episode 3 to transform the results
    back to the GD-1 frame.
    """)
    return


@app.cell
def _(candidate_gaia_table):
    candidate_gaia_df = make_dataframe(candidate_gaia_table)
    return (candidate_gaia_df,)


@app.cell
def _(candidate_gaia_df):
    plot_pm_selection(candidate_gaia_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We are starting to see GD-1 more clearly.

    In the next episode, we will use photometry data from Pan-STARRS to do
    a second round of filtering, and see if we can replicate Figure 1 from
    the original paper.

    ## Summary

    In this episode, we improved the selection process by writing a more complex
    query that uses the database to select stars based on proper motion.
    This process requires more computation on the Gaia server, but then
    we are able to either:

    1. Search the same region and download less data, or
    2. Search a larger region while still downloading a manageable amount of data.

    In the next episode, we will learn about the database `JOIN` operation.

    **Key point:** When possible, 'move the computation to the data'; that is, do
    as much of the work as possible on the database server before downloading the data.
    """)
    return

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
    # Coordinate Transformations

    In the previous episode, we wrote ADQL queries and used them to select
    and download data from the Gaia server. In this episode, we will write
    a query to select stars from a particular region of the sky.

    We'll start with an example that does a "cone search", then select stars
    in the vicinity of GD-1 by:

    - Using `Quantity` objects to represent measurements with units
    - Using Astropy to convert coordinates from one frame to another
    - Using ADQL keywords `POLYGON`, `CONTAINS`, and `POINT` to select stars
      that fall within a polygonal region
    - Submitting a query and downloading the results
    - Storing the results in a FITS file

    ## Working with Units

    Astropy provides tools for including units explicitly in computations.
    To use Astropy units, we import them like this:
    """)
    return


@app.cell
def _():
    import astropy.units as u
    return (u,)


@app.cell
def _(u):
    # Create a quantity by multiplying a value by a unit
    angle = 10 * u.degree
    angle
    return (angle,)


@app.cell
def _(angle, u):
    # Convert to other units using the .to() method
    angle_arcmin = angle.to(u.arcmin)
    angle_arcmin
    return (angle_arcmin,)


@app.cell
def _(angle, u):
    # Astropy converts compatible units automatically when adding
    result = angle + 30 * u.arcmin
    result
    return (result,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    Create a quantity that represents 5 arcminutes and assign it to a
    variable called `radius`. Then convert it to degrees.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    ```python
    radius = 5 * u.arcmin
    print(radius)

    radius.to(u.degree)
    ```
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Selecting a Region

    One of the most common ways to restrict a query is to select stars in
    a particular region of the sky. Here is an ADQL query that selects
    objects in a circular region (a "cone search"):
    """)
    return


@app.cell
def _():
    cone_query = """SELECT
TOP 10
source_id
FROM gaiadr2.gaia_source
WHERE 1=CONTAINS(
  POINT(ra, dec),
  CIRCLE(88.8, 7.4, 0.08333333))
"""
    return (cone_query,)


@app.cell
def _(cone_query):
    from astroquery.gaia import Gaia

    cone_job = Gaia.launch_job(cone_query)
    cone_results = cone_job.get_results()
    cone_results
    return (Gaia, cone_job, cone_results)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    When debugging queries, you can use `TOP` to limit the size of the
    results, but then you still don't know how big the full result would be.

    An alternative is to use `COUNT`, which asks for the number of rows that
    would be selected without returning them.

    In the previous query, replace `TOP 10 source_id` with `COUNT(source_id)`
    and run the query again. How many stars has Gaia identified in the cone
    we searched?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    ```python
    count_cone_query = """SELECT
    COUNT(source_id)
    FROM gaiadr2.gaia_source
    WHERE 1=CONTAINS(
      POINT(ra, dec),
      CIRCLE(88.8, 7.4, 0.08333333))
    """

    count_cone_job = Gaia.launch_job(count_cone_query)
    count_cone_results = count_cone_job.get_results()
    count_cone_results
    # Result: 594 stars
    ```
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Transforming Coordinates

    Astropy provides a `SkyCoord` object that represents sky coordinates
    relative to a specified reference frame.

    The following example creates a `SkyCoord` object for Betelgeuse in
    the ICRS frame (International Celestial Reference System).
    """)
    return


@app.cell
def _(u):
    from astropy.coordinates import SkyCoord

    ra = 88.8 * u.degree
    dec = 7.4 * u.degree
    coord_icrs = SkyCoord(ra=ra, dec=dec, frame='icrs')
    coord_icrs
    return (SkyCoord, coord_icrs, dec, ra)


@app.cell
def _(coord_icrs):
    # Transform to Galactic coordinates
    coord_galactic = coord_icrs.transform_to('galactic')
    coord_galactic
    return (coord_galactic,)


@app.cell
def _(mo):
    mo.md(r"""
    To transform to and from GD-1 coordinates, we use the `GD1Koposov10`
    frame. This is "a Heliocentric spherical coordinate system defined by
    the orbit of the GD-1 stream", where phi1 is along the stream direction
    and phi2 is perpendicular.
    """)
    return


@app.cell
def _():
    from gd1 import GD1Koposov10

    gd1_frame = GD1Koposov10()
    gd1_frame
    return (GD1Koposov10, gd1_frame)


@app.cell
def _(coord_icrs, gd1_frame):
    coord_gd1 = coord_icrs.transform_to(gd1_frame)
    coord_gd1
    return (coord_gd1,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (10 minutes)

    Find the location of GD-1 in ICRS coordinates.

    1. Create a `SkyCoord` object at 0°, 0° in the GD-1 frame.
    2. Transform it to the ICRS frame.

    Hint: Because ICRS is a standard frame, it is built into Astropy.
    You can specify it by name, `icrs`.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
    ```python
    origin_gd1 = SkyCoord(0*u.degree, 0*u.degree, frame=gd1_frame)

    origin_gd1.transform_to('icrs')
    ```

    The origin of the GD-1 frame maps to `ra=200`, exactly, in ICRS. That is by design.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Selecting a Rectangle

    We will select stars in a small rectangle near the center of GD-1,
    from -55 to -45 degrees phi1 and -8 to 4 degrees phi2.

    The `make_rectangle` function takes lower and upper bounds and returns
    the corners of a rectangle starting with the lower left corner and
    working clockwise.
    """)
    return


@app.cell
def _(u):
    phi1_min = -55 * u.degree
    phi1_max = -45 * u.degree
    phi2_min = -8 * u.degree
    phi2_max = 4 * u.degree
    return (phi1_max, phi1_min, phi2_max, phi2_min)


@app.cell
def _():
    def make_rectangle(x1, x2, y1, y2):
        """Return the corners of a rectangle."""
        xs = [x1, x1, x2, x2, x1]
        ys = [y1, y2, y2, y1, y1]
        return xs, ys
    return (make_rectangle,)


@app.cell
def _(make_rectangle, phi1_max, phi1_min, phi2_max, phi2_min):
    phi1_rect, phi2_rect = make_rectangle(
        phi1_min, phi1_max, phi2_min, phi2_max)
    return (phi1_rect, phi2_rect)


@app.cell
def _(SkyCoord, gd1_frame, phi1_rect, phi2_rect):
    corners = SkyCoord(phi1=phi1_rect, phi2=phi2_rect, frame=gd1_frame)
    corners
    return (corners,)


@app.cell
def _(corners):
    # Convert the rectangle corners from GD-1 frame to ICRS
    # (a rectangle in GD-1 frame becomes a non-rectangular polygon in ICRS)
    corners_icrs = corners.transform_to('icrs')
    corners_icrs
    return (corners_icrs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Defining a Polygon

    In order to use this polygon as part of an ADQL query, we have to
    convert it to a string with a comma-separated list of coordinates.

    We'll write a helper function to do this conversion.
    """)
    return


@app.cell
def _():
    def skycoord_to_string(skycoord):
        """Convert a one-dimensional list of SkyCoord to string for Gaia's query format."""
        corners_list_str = skycoord.to_string()
        corners_single_str = ' '.join(corners_list_str)
        return corners_single_str.replace(' ', ', ')
    return (skycoord_to_string,)


@app.cell
def _(corners_icrs, skycoord_to_string):
    sky_point_list = skycoord_to_string(corners_icrs)
    sky_point_list
    return (sky_point_list,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Assembling the Query

    Now we are ready to assemble our query to get all of the stars in
    the Gaia catalog that are in the small rectangle we defined.

    First, let's test with `TOP 10` to make sure the query works.
    """)
    return


@app.cell
def _():
    columns = 'source_id, ra, dec, pmra, pmdec, parallax'
    return (columns,)


@app.cell
def _(columns, sky_point_list):
    polygon_top10query_base = """SELECT
TOP 10
{columns}
FROM gaiadr2.gaia_source
WHERE parallax < 1
  AND bp_rp BETWEEN -0.75 AND 2
  AND 1 = CONTAINS(POINT(ra, dec),
                   POLYGON({sky_point_list}))
"""
    polygon_top10query = polygon_top10query_base.format(
        columns=columns,
        sky_point_list=sky_point_list)
    print(polygon_top10query)
    return (polygon_top10query, polygon_top10query_base)


@app.cell
def _(Gaia, polygon_top10query):
    polygon_top10query_job = Gaia.launch_job_async(polygon_top10query)
    polygon_top10query_results = polygon_top10query_job.get_results()
    polygon_top10query_results
    return (polygon_top10query_job, polygon_top10query_results)


@app.cell
def _(mo):
    mo.md(r"""
    Now we can remove `TOP 10` and run the full query. This will take a
    little longer since it returns more than 100,000 stars.
    """)
    return


@app.cell
def _(columns, sky_point_list):
    polygon_query_base = """SELECT
{columns}
FROM gaiadr2.gaia_source
WHERE parallax < 1
  AND bp_rp BETWEEN -0.75 AND 2
  AND 1 = CONTAINS(POINT(ra, dec),
                   POLYGON({sky_point_list}))
"""
    polygon_query = polygon_query_base.format(
        columns=columns,
        sky_point_list=sky_point_list)
    print(polygon_query)
    return (polygon_query, polygon_query_base)


@app.cell
def _(Gaia, polygon_query):
    polygon_job = Gaia.launch_job_async(polygon_query)
    print(polygon_job)
    return (polygon_job,)


@app.cell
def _(polygon_job):
    polygon_results = polygon_job.get_results()
    len(polygon_results)
    return (polygon_results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Saving Results

    This is the set of stars we will work with in the next step. Since
    we have a substantial dataset now, this is a good time to save it.

    Astropy `Table` objects provide `write`, which writes the table to disk.
    """)
    return


@app.cell
def _(polygon_results):
    from os.path import getsize

    filename = 'gd1_results.fits'
    polygon_results.write(filename, overwrite=True)

    # MB is defined as 1024 * 1024 bytes
    MB = 1024 * 1024
    getsize(filename) / MB
    return (MB, filename, getsize)


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    In this notebook, we composed more complex queries to select stars
    within a polygonal region of the sky. Then we downloaded the results
    and saved them in a FITS file.

    Key points:
    - For measurements with units, use `Quantity` objects that represent units explicitly
    - Use the `format` function to compose queries; it is often faster and less error-prone
    - Develop queries incrementally: start with something simple, test it, and add a little bit at a time
    - Once you have a query working, save the data in a local file
    """)
    return


if __name__ == "__main__":
    app.run()

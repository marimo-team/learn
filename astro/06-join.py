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
    # Join

    The next step in our analysis is to select candidate stars based on photometry data.

    In red is a stellar isochrone, showing where we expect the stars in GD-1 to fall
    based on the metallicity and age of their original globular cluster.

    By selecting stars in the shaded area, we can further distinguish the main sequence
    of GD-1 from younger background stars.

    ## Outline

    1. We will reload the candidate stars we identified in the previous episode.

    2. Then we will run a query on the Gaia server that uploads the table of candidates
       and uses a `JOIN` operation to select photometry data for the candidate stars.

    3. We will write the results to a file for use in the next episode.
    """)
    return


@app.cell
def _():
    from astroquery.gaia import Gaia
    import pandas as pd

    from episode_functions import *
    return Gaia, pd


@app.cell
def _(pd):
    filename = 'gd1_data.hdf'
    point_series = pd.read_hdf(filename, 'point_series')
    sky_point_list = point_series['sky_point_list']
    pmra_min = point_series['pmra_min']
    pmra_max = point_series['pmra_max']
    pmdec_min = point_series['pmdec_min']
    pmdec_max = point_series['pmdec_max']
    point_series
    return (
        filename,
        pmra_max,
        pmra_min,
        pmdec_max,
        pmdec_min,
        point_series,
        sky_point_list,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Getting photometry data

    The Gaia dataset contains some photometry data, including the variable `bp_rp`,
    which contains BP-RP color (the difference in mean flux between the BP and RP bands).
    We use this variable to select stars with `bp_rp` between -0.75 and 2, which excludes
    many class M dwarf stars.

    But we can do better than that. Assuming GD-1 is a globular cluster, all of the stars
    formed at the same time from the same material, so the stars' photometric properties
    should be consistent with a single isochrone in a color magnitude diagram.
    We can use photometric color and apparent magnitude to select stars with the age and
    metal richness we expect in GD-1.

    Conveniently, the Gaia server provides data from Pan-STARRS as a table in the same
    database we have been using, so we can access it by making ADQL queries.

    **A caveat about matching stars between catalogs:** In general, choosing a star from
    the Gaia catalog and finding the corresponding star in the Pan-STARRS catalog is not
    easy. Fortunately, the Gaia database includes cross-matching tables that suggest a
    best neighbor in the Pan-STARRS catalog for many stars in the Gaia catalog.

    **Note on British spelling:** The Gaia database uses the British spelling of
    "neighbour" (with a "u"). Do not forget to include it in your table names.

    ## The best neighbor table

    Here is the metadata for `panstarrs1_best_neighbour`.
    """)
    return


@app.cell
def _(Gaia):
    ps_best_neighbour_meta = Gaia.load_table('gaiadr2.panstarrs1_best_neighbour')
    return (ps_best_neighbour_meta,)


@app.cell
def _(ps_best_neighbour_meta):
    print(ps_best_neighbour_meta)
    return


@app.cell
def _(ps_best_neighbour_meta):
    for column in ps_best_neighbour_meta.columns:
        print(column.name)
    return (column,)


@app.cell
def _(mo):
    mo.md(r"""
    The ones we will use are:

    - `source_id`, which we will match up with `source_id` in the Gaia table.
    - `best_neighbour_multiplicity`, which indicates how many sources in Pan-STARRS
      are matched with the same probability to this source in Gaia.
    - `number_of_mates`, which indicates the number of *other* sources in Gaia that
      are matched with the same source in Pan-STARRS.
    - `original_ext_source_id`, which we will match up with `obj_id` in the Pan-STARRS table.

    Ideally, `best_neighbour_multiplicity` should be 1 and `number_of_mates` should be 0;
    in that case, there is a one-to-one match between the source in Gaia and the
    corresponding source in Pan-STARRS.

    Here is a query that selects these columns and returns the first 5 rows.
    """)
    return


@app.cell
def _(Gaia):
    ps_best_neighbour_query = """SELECT
    TOP 5
    source_id, best_neighbour_multiplicity, number_of_mates, original_ext_source_id
    FROM gaiadr2.panstarrs1_best_neighbour
    """
    ps_best_neighbour_job = Gaia.launch_job_async(ps_best_neighbour_query)
    ps_best_neighbour_results = ps_best_neighbour_job.get_results()
    ps_best_neighbour_results
    return (
        ps_best_neighbour_job,
        ps_best_neighbour_query,
        ps_best_neighbour_results,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## The Pan-STARRS table

    Now that we know the Pan-STARRS `obj_id`, we are ready to match this to the
    photometry in the `panstarrs1_original_valid` table.
    """)
    return


@app.cell
def _(Gaia):
    ps_valid_meta = Gaia.load_table('gaiadr2.panstarrs1_original_valid')
    print(ps_valid_meta)
    return (ps_valid_meta,)


@app.cell
def _(ps_valid_meta):
    for col in ps_valid_meta.columns:
        print(col.name)
    return (col,)


@app.cell
def _(mo):
    mo.md(r"""
    The ones we will use are:

    - `obj_id`, which we will match up with `original_ext_source_id` in the best neighbor table.
    - `g_mean_psf_mag`, which contains mean magnitude from the `g` filter.
    - `i_mean_psf_mag`, which contains mean magnitude from the `i` filter.

    Here is a query that selects these variables and returns the first 5 rows.
    """)
    return


@app.cell
def _(Gaia):
    ps_valid_query = """SELECT
    TOP 5
    obj_id, g_mean_psf_mag, i_mean_psf_mag
    FROM gaiadr2.panstarrs1_original_valid
    """
    ps_valid_job = Gaia.launch_job_async(ps_valid_query)
    ps_valid_results = ps_valid_job.get_results()
    ps_valid_results
    return ps_valid_job, ps_valid_query, ps_valid_results


@app.cell
def _(mo):
    mo.md(r"""
    ## Joining tables

    The following figure shows how these tables are related.

    - The first `JOIN` operation takes each `source_id` in the Gaia table and finds the
      same value of `source_id` in the best neighbor table.

    - The second `JOIN` operation takes each `original_ext_source_id` in the best neighbor
      table and finds the same value of `obj_id` in the PanSTARRS photometry table.

    We are going to start with a simplified version of what we want to do until we are
    sure we are joining the tables correctly, then we will slowly add more layers of
    complexity, checking at each stage that our query still works.

    As a starting place, we will go all the way back to the cone search from episode 2.
    """)
    return


@app.cell
def _(Gaia):
    test_cone_query = """SELECT
    TOP 10
    source_id
    FROM gaiadr2.gaia_source
    WHERE 1=CONTAINS(
      POINT(ra, dec),
      CIRCLE(88.8, 7.4, 0.08333333))
    """
    test_cone_job = Gaia.launch_job(test_cone_query)
    test_cone_results = test_cone_job.get_results()
    test_cone_results
    return test_cone_job, test_cone_query, test_cone_results


@app.cell
def _(mo):
    mo.md(r"""
    Now we can start adding features. First, we will replace `source_id` with the
    format specifier `columns` so that we can alter what columns we want to return
    without having to modify our base query.
    """)
    return


@app.cell
def _():
    cone_base_query = """SELECT
    {columns}
    FROM gaiadr2.gaia_source
    WHERE 1=CONTAINS(
      POINT(ra, dec),
      CIRCLE(88.8, 7.4, 0.08333333))
    """
    return (cone_base_query,)


@app.cell
def _(Gaia, cone_base_query):
    columns_cone = 'source_id, ra, dec, pmra, pmdec'

    cone_query = cone_base_query.format(columns=columns_cone)
    print(cone_query)
    return columns_cone, cone_query


@app.cell
def _(Gaia, cone_query):
    cone_job = Gaia.launch_job_async(cone_query)
    cone_results = cone_job.get_results()
    cone_results
    return cone_job, cone_results


@app.cell
def _(mo):
    mo.md(r"""
    ## Adding the best neighbor table

    Now we are ready for the first join. The join operation requires two clauses:

    - `JOIN` specifies the name of the table we want to join with, and
    - `ON` specifies how we will match up rows between the tables.

    In this example, we join with `gaiadr2.panstarrs1_best_neighbour AS best`, which
    means we can refer to the best neighbor table with the abbreviated name `best`.
    Similarly, we will be referring to the `gaiadr2.gaia_source` table by the
    abbreviated name `gaia`.

    The `ON` clause indicates that we will match up the `source_id` column from the
    Gaia table with the `source_id` column from the best neighbor table.
    """)
    return


@app.cell
def _():
    neighbours_base_query = """SELECT
    {columns}
    FROM gaiadr2.gaia_source AS gaia
    JOIN gaiadr2.panstarrs1_best_neighbour AS best
      ON gaia.source_id = best.source_id
    WHERE 1=CONTAINS(
      POINT(gaia.ra, gaia.dec),
      CIRCLE(88.8, 7.4, 0.08333333))
    """
    return (neighbours_base_query,)


@app.cell
def _(Gaia, neighbours_base_query):
    column_list_neighbours = ['gaia.source_id',
                   'gaia.ra',
                   'gaia.dec',
                   'gaia.pmra',
                   'gaia.pmdec',
                   'best.best_neighbour_multiplicity',
                   'best.number_of_mates',
                  ]
    columns_neighbours = ', '.join(column_list_neighbours)

    neighbours_query = neighbours_base_query.format(columns=columns_neighbours)
    print(neighbours_query)
    return column_list_neighbours, columns_neighbours, neighbours_query


@app.cell
def _(Gaia, neighbours_query):
    neighbours_job = Gaia.launch_job_async(neighbours_query)
    neighbours_results = neighbours_job.get_results()
    neighbours_results
    return neighbours_job, neighbours_results


@app.cell
def _(mo):
    mo.md(r"""
    This result has fewer rows than the previous result. That is because there are
    sources in the Gaia table with no corresponding source in the Pan-STARRS table.

    By default, the result of the join only includes rows where the same `source_id`
    appears in both tables. This default is called an "inner" join because the results
    include only the intersection of the two tables.

    ## Adding the Pan-STARRS table
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (15 minutes)

    Now we are ready to bring in the Pan-STARRS table. Starting with the previous
    query, add a second `JOIN` clause that joins with `gaiadr2.panstarrs1_original_valid`,
    gives it the abbreviated name `ps`, and matches `original_ext_source_id` from the
    best neighbor table with `obj_id` from the Pan-STARRS table.

    Add `g_mean_psf_mag` and `i_mean_psf_mag` to the column list, and run the query.
    The result should contain 490 rows and 9 columns.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
join_solution_query_base = """SELECT
{columns}
FROM gaiadr2.gaia_source as gaia
JOIN gaiadr2.panstarrs1_best_neighbour as best
  ON gaia.source_id = best.source_id
JOIN gaiadr2.panstarrs1_original_valid as ps
  ON best.original_ext_source_id = ps.obj_id
WHERE 1=CONTAINS(
  POINT(gaia.ra, gaia.dec),
  CIRCLE(88.8, 7.4, 0.08333333))
"""

column_list = ['gaia.source_id',
               'gaia.ra',
               'gaia.dec',
               'gaia.pmra',
               'gaia.pmdec',
               'best.best_neighbour_multiplicity',
               'best.number_of_mates',
               'ps.g_mean_psf_mag',
               'ps.i_mean_psf_mag']

columns = ', '.join(column_list)

join_solution_query = join_solution_query_base.format(columns=columns)
print(join_solution_query)

join_solution_job = Gaia.launch_job_async(join_solution_query)
join_solution_results = join_solution_job.get_results()
join_solution_results
```
        """)
    })
    return


@app.cell
def _():
    column_list = ['gaia.source_id',
                   'gaia.ra',
                   'gaia.dec',
                   'gaia.pmra',
                   'gaia.pmdec',
                   'best.best_neighbour_multiplicity',
                   'best.number_of_mates',
                   'ps.g_mean_psf_mag',
                   'ps.i_mean_psf_mag']
    return (column_list,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Selecting by coordinates and proper motion

    We are now going to replace the cone search with the GD-1 selection that we built
    in previous episodes. We will start by making sure that our previous query works,
    then add in the `JOIN`.

    Here is `candidate_coord_pm_query_base` from the previous episode.
    """)
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
def _(
    Gaia,
    candidate_coord_pm_query_base,
    pmra_max,
    pmra_min,
    pmdec_max,
    pmdec_min,
    sky_point_list,
):
    columns_verify = 'source_id, ra, dec, pmra, pmdec'

    candidate_coord_pm_query = candidate_coord_pm_query_base.format(
        columns=columns_verify,
        sky_point_list=sky_point_list,
        pmra_min=pmra_min,
        pmra_max=pmra_max,
        pmdec_min=pmdec_min,
        pmdec_max=pmdec_max)

    print(candidate_coord_pm_query)
    return candidate_coord_pm_query, columns_verify


@app.cell
def _(Gaia, candidate_coord_pm_query):
    candidate_coord_pm_job = Gaia.launch_job_async(candidate_coord_pm_query)
    candidate_coord_pm_results = candidate_coord_pm_job.get_results()
    candidate_coord_pm_results
    return candidate_coord_pm_job, candidate_coord_pm_results


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (15 minutes)

    Create a new query base called `candidate_join_query_base` that combines the `WHERE`
    clauses from the previous query with the `JOIN` clauses for the best neighbor and
    Pan-STARRS tables.
    Format the query base using the column names in `column_list`, and call the result
    `candidate_join_query`.

    Hint: Make sure you use qualified column names everywhere!

    Run your query and download the results. The table you get should have 4300 rows
    and 9 columns.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
candidate_join_query_base = """
SELECT
{columns}
FROM gaiadr2.gaia_source as gaia
JOIN gaiadr2.panstarrs1_best_neighbour as best
  ON gaia.source_id = best.source_id
JOIN gaiadr2.panstarrs1_original_valid as ps
  ON best.original_ext_source_id = ps.obj_id
WHERE parallax < 1
  AND bp_rp BETWEEN -0.75 AND 2
  AND 1 = CONTAINS(POINT(gaia.ra, gaia.dec),
                   POLYGON({sky_point_list}))
  AND gaia.pmra BETWEEN {pmra_min} AND  {pmra_max}
  AND gaia.pmdec BETWEEN {pmdec_min} AND {pmdec_max}
"""

columns = ', '.join(column_list)

candidate_join_query = candidate_join_query_base.format(
    columns=columns,
    sky_point_list=sky_point_list,
    pmra_min=pmra_min,
    pmra_max=pmra_max,
    pmdec_min=pmdec_min,
    pmdec_max=pmdec_max)
print(candidate_join_query)

candidate_join_job = Gaia.launch_job_async(candidate_join_query)
candidate_table = candidate_join_job.get_results()
candidate_table
```
        """)
    })
    return


@app.cell
def _(
    Gaia,
    column_list,
    pmra_max,
    pmra_min,
    pmdec_max,
    pmdec_min,
    sky_point_list,
):
    candidate_join_query_base = """
SELECT
{columns}
FROM gaiadr2.gaia_source as gaia
JOIN gaiadr2.panstarrs1_best_neighbour as best
  ON gaia.source_id = best.source_id
JOIN gaiadr2.panstarrs1_original_valid as ps
  ON best.original_ext_source_id = ps.obj_id
WHERE parallax < 1
  AND bp_rp BETWEEN -0.75 AND 2
  AND 1 = CONTAINS(POINT(gaia.ra, gaia.dec),
                   POLYGON({sky_point_list}))
  AND gaia.pmra BETWEEN {pmra_min} AND  {pmra_max}
  AND gaia.pmdec BETWEEN {pmdec_min} AND {pmdec_max}
"""

    columns_join = ', '.join(column_list)

    candidate_join_query = candidate_join_query_base.format(
        columns=columns_join,
        sky_point_list=sky_point_list,
        pmra_min=pmra_min,
        pmra_max=pmra_max,
        pmdec_min=pmdec_min,
        pmdec_max=pmdec_max)
    print(candidate_join_query)

    candidate_join_job = Gaia.launch_job_async(candidate_join_query)
    candidate_table = candidate_join_job.get_results()
    candidate_table
    return (
        candidate_join_job,
        candidate_join_query,
        candidate_join_query_base,
        candidate_table,
        columns_join,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Checking the match

    To get more information about the matching process, we can inspect
    `best_neighbour_multiplicity`, which indicates for each star in Gaia
    how many stars in Pan-STARRS are equally likely matches.
    """)
    return


@app.cell
def _(candidate_table, pd):
    candidate_table['best_neighbour_multiplicity']
    return


@app.cell
def _(candidate_table, pd):
    multiplicity = pd.Series(candidate_table['best_neighbour_multiplicity'])
    multiplicity.describe()
    return (multiplicity,)


@app.cell
def _(mo):
    mo.md(r"""
    In fact, `1` is the only value in the `Series`, so every candidate star has a
    single best match.

    Similarly, `number_of_mates` indicates the number of *other* stars in Gaia that
    match with the same star in Pan-STARRS.
    """)
    return


@app.cell
def _(candidate_table, pd):
    mates = pd.Series(candidate_table['number_of_mates'])
    mates.describe()
    return (mates,)


@app.cell
def _(mo):
    mo.md(r"""
    All values in this column are `0`, which means that for each match we found in
    Pan-STARRS, there are no other stars in Gaia that also match.

    ## Saving the DataFrame

    We can make a `DataFrame` from our Astropy `Table` and save our results so we
    can pick up where we left off without running this query again.
    """)
    return


@app.cell
def _(candidate_table):
    candidate_df = make_dataframe(candidate_table)
    return (candidate_df,)


@app.cell
def _(candidate_df, filename):
    candidate_df.to_hdf(filename, key='candidate_df')
    return


@app.cell
def _(filename):
    from os.path import getsize

    # 1 MB = 1024 * 1024 bytes
    MB = 1024 * 1024
    getsize(filename) / MB
    return MB, getsize


@app.cell
def _(mo):
    mo.md(r"""
    ## Another file format - CSV

    Pandas can write a variety of other formats. One other important one is CSV
    (comma-separated values), which is a plain-text format that can be read and
    written by pretty much any tool that works with data.

    However, it has an important limitation: some information about the data gets
    lost in translation, notably the data types. Also, CSV files tend to be big,
    and slow to read and write.
    """)
    return


@app.cell
def _(MB, candidate_df, getsize):
    candidate_df.to_csv('gd1_data.csv')
    getsize('gd1_data.csv') / MB
    return


@app.cell
def _(pd):
    read_back_csv = pd.read_csv('gd1_data.csv')
    return (read_back_csv,)


@app.cell
def _(candidate_df):
    candidate_df.head(3)
    return


@app.cell
def _(read_back_csv):
    read_back_csv.head(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The CSV file contains the names of the columns, but not the data types.
    Additionally, notice that the index in `candidate_df` has become an unnamed
    column in `read_back_csv` and a new index has been created. The Pandas functions
    for writing and reading CSV files provide options to avoid that problem, but
    this is an example of the kind of thing that can go wrong with CSV files.

    ## Summary

    In this episode, we used database `JOIN` operations to select photometry data
    for the stars we've identified as candidates to be in GD-1.

    In the next episode, we will use this data for a second round of selection,
    identifying stars that have photometry data consistent with GD-1.

    **Key points:**

    - Use `JOIN` operations to combine data from multiple tables in a database,
      using some kind of identifier to match up records from one table with records
      from another. This is another example of a practice we saw in the previous
      notebook, moving the computation to the data.
    - For most applications, saving data in FITS or HDF5 is better than CSV. FITS
      and HDF5 are binary formats, so the files are usually smaller, and they store
      metadata, so you don't lose anything when you read the file back.
    - On the other hand, CSV is a 'least common denominator' format; that is, it can
      be read by practically any application that works with data.
    """)
    return

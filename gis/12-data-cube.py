# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.9.0",
#     "rioxarray",
#     "numpy",
#     "matplotlib",
#     "geopandas",
#     "xarray",
#     "dask",
#     "pystac-client",
#     "odc-stac",
# ]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data cubes with ODC-STAC
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions

    - Can I mosaic tiled raster datasets when my area of interest spans multiple files?
    - Can I stack raster datasets that cover the same area along the time dimension in order to explore temporal
      changes of some quantities?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objectives

    - ODC-STAC allows you to work with raster datasets spanning multiple files as if they were a single
      multi-dimensional object (a "data cube").
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    In the previous episodes we worked with satellite images with a fixed boundary on how they have been
    collected, however in many cases you would want to have an image that covers your area of interest which
    often does not align with boundaries of the collected images. If the phenomena you are interested in covers
    two images you could manually mosaic them, but sometimes you are interested in multiple images that overlap.

    ODC-STAC offers functionality that allows you to get a mosaiced image based on a bounding box or a polygon
    containing the area of interest. In this lesson we show how
    [odc-stac](https://odc-stac.readthedocs.io/en/latest/?badge=latest) can be employed to re-tile and stack
    satellite images in what are sometimes referred to as "data cubes".
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a data cube with ODC-STAC

    As you might have noticed in the previous episodes, the satellite images we have used until now do actually
    not cover the whole island of Rhodes. They miss the southern part of the island. Using ODC-STAC we can obtain
    an image for the whole island. We use the administrative boundary of Rhodes to define our area of interest
    (AoI). This way we are sure to have the whole island.

    More important, using ODC-STAC you can also load multiple images (lazy) into one datacube allowing you to
    perform all kind of interesting analyses as will be demonstrated below.

    But first we need to upload the geometry of Rhodes. To do so we use geopandas and load the geometry we
    previously stored in a geopackage.
    """)
    return


@app.cell
def _():
    import geopandas
    rhodes = geopandas.read_file('rhodes.gpkg')
    bbox = rhodes.total_bounds
    return (bbox, geopandas, rhodes)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we search for satellite images that cover our AoI (i.e. Rhodes) in the
    [Sentinel-2 L2A](https://radiantearth.github.io/stac-browser/#/external/earth-search.aws.element84.com/v1/collections/sentinel-2-l2a)
    collection that is indexed in the
    [Earth Search STAC API](https://radiantearth.github.io/stac-browser/#/external/earth-search.aws.element84.com/v1/).
    Since we are interested in the period right before and after the wild fire we include as dates the 1st of July
    until the 31st of August 2023:
    """)
    return


@app.cell
def _(bbox):
    import pystac_client

    api_url = "https://earth-search.aws.element84.com/v1"
    collection_id = "sentinel-2-c1-l2a"

    client = pystac_client.Client.open(api_url)
    search = client.search(
        collections=[collection_id],
        datetime="2023-07-01/2023-08-31",
        bbox=bbox
    )

    item_collection = search.item_collection()
    return (api_url, client, collection_id, item_collection, pystac_client, search)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [odc-stac](https://odc-stac.readthedocs.io/en/latest/?badge=latest) can ingest directly our search results
    and create a Xarray DataSet object from the STAC metadata that are present in the `item_collection`. By
    specifying `groupby='solar_day'`, odc-stac automatically groups and merges images corresponding to the same
    date of acquisition. `chunks={...}` sets up the resulting data cube using Dask arrays, thus enabling lazy
    loading (and further operations). `use_overviews=True` tells odc-stac to directly load lower-resolution
    versions of the images from the overviews, if these are available in Cloud Optimized GeoTIFFs (COGs). We set
    the resolution of the data cube using the `resolution` argument, and define the AoI using the bounding box
    (`bbox`). We decided to set the resolution to 20 in order to limit the size of the images a bit.
    """)
    return


@app.cell
def _(item_collection, rhodes):
    import odc.stac
    ds = odc.stac.load(
        item_collection,
        groupby='solar_day',
        chunks={'x': 2048, 'y': 2048},
        use_overviews=True,
        resolution=20,
        bbox=rhodes.total_bounds,
    )
    return (ds, odc)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    odc-stac builds a data cube representation from all the relevant files linked in `item_collection` as a
    Xarray DataSet. Let us have a look at it:
    """)
    return


@app.cell
def _(ds):
    print(ds)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Working with the data cube

    Like we did in the previous episode, let us calculate the NDVI for our study area. To do so we need to focus
    on the variables: the red band (`red`), the near infrared band (`nir`) and the scene classification map
    (`scl`). We will use the former two to calculate the NDVI for the AoI. The latter, we use as
    [a classification mask](https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-SceneClassification(SC)S2-Processing-Scene-Classificationtrue)
    provided together with Sentinel-2 L2A products. In this mask, each pixel is classified according to a set
    of labels.

    First we define the bands that we are interested in:
    """)
    return


@app.cell
def _(ds):
    red = ds['red']
    nir = ds['nir']
    scl = ds['scl']
    return (nir, red, scl)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we will use the mask to drop pixels that are labeled as clouds and water. For this we use the
    classification map to mask out pixels recognized by the
    [Sentinel-2 processing algorithm](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/)
    as cloud or water:
    """)
    return


@app.cell
def _(nir, red, scl):
    # generate mask ("True" for pixel being cloud or water)
    mask = scl.isin([
        3,  # CLOUD_SHADOWS
        6,  # WATER
        8,  # CLOUD_MEDIUM_PROBABILITY
        9,  # CLOUD_HIGH_PROBABILITY
        10  # THIN_CIRRUS
    ])
    red_masked = red.where(~mask)
    nir_masked = nir.where(~mask)
    return (mask, nir_masked, red_masked)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then, we calculate the NDVI:
    """)
    return


@app.cell
def _(nir_masked, red_masked):
    ndvi = (nir_masked - red_masked) / (nir_masked + red_masked)
    return (ndvi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can visualize the calculated NDVI for the AoI at two given dates (before and after the wildfires) by
    selecting the date:
    """)
    return


@app.cell
def _(ndvi):
    ndvi_before = ndvi.sel(time="2023-07-13")
    ndvi_before.plot()
    return (ndvi_before,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![NDVI before the wildfire](/public/E12/NDVI-before.png)
    """)
    return


@app.cell
def _(ndvi):
    ndvi_after = ndvi.sel(time="2023-08-27")
    ndvi_after.plot()
    return (ndvi_after,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![NDVI after the wildfire](/public/E12/NDVI-after.png)

    Another feature of having the data available in a datacube is that you can for instance query multiple
    layers. If you want to see how the NDVI changed over time for a specific point you can do the following.
    Let us first define a point in the region where we know it was affected by the wildfire. To check that it
    is after the fire we plot it:
    """)
    return


@app.cell
def _(ndvi_after):
    import matplotlib.pyplot as plt
    x = 585_000
    y = 3_995_000

    fig, ax = plt.subplots()
    ndvi_after.plot(ax=ax)
    ax.scatter(x, y, marker="o", c="k")
    return (ax, fig, plt, x, y)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![NDVI plot with selected point](/public/E12/NDVI-after_point.png)

    Now let us extract the NDVI values computed at that point for the full time series:
    """)
    return


@app.cell
def _(ndvi, x, y):
    ndvi_xy_1 = ndvi.sel(x=x, y=y, method="nearest")
    print(ndvi_xy_1)
    return (ndvi_xy_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now trigger computation. Note that we run this in parallel (probably not much of an effect here, but
    definitely helpful for larger calculations):
    """)
    return


@app.cell
def _(ndvi_xy_1):
    import time
    _t0 = time.time()
    ndvi_xy_2 = ndvi_xy_1.compute(scheduler="threads", num_workers=4)
    _t1 = time.time()
    print(f"Wall time: {_t1 - _t0:.1f} s")
    return (ndvi_xy_2, time)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The result is a time series representing the NDVI value computed for the selected point for all the available
    scenes in the time range. We drop the NaN values, and plot the final result:
    """)
    return


@app.cell
def _(ndvi_xy_2):
    ndvi_xy_2.dropna(dim="time").plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![NDVI time series](/public/E12/NDVI-time-series.png)
    """)
    return


if __name__ == "__main__":
    app.run()

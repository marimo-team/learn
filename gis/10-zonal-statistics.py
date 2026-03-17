# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.9.0",
#     "rioxarray",
#     "rasterio",
#     "numpy",
#     "matplotlib",
#     "geopandas",
#     "xarray",
#     "xarray-spatial",
# ]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Calculating Zonal Statistics on Rasters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions

    - How to compute raster statistics on different zones delineated by vector data?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objectives

    - Extract zones from the vector dataset
    - Convert vector data to raster
    - Calculate raster statistics over zones
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    Statistics on predefined zones of the raster data are commonly used for analysis and to better understand the
    data. These zones are often provided within a single vector dataset, identified by certain vector attributes.
    For example, in the previous episodes, we defined infrastructure regions and built-up regions on Rhodes Island
    as polygons. Each region can be respectively identified as a "zone", resulting in two zones. One can evaluate
    the effect of the wild fire on the two zones by calculating the zonal statistics.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading

    We have created `assets.gpkg` in Episode "Vector data in Python", which contains the infrastructure regions
    and built-up regions. We also calculated the burned index in Episode "Raster Calculations in Python" and saved
    it in `burned.tif`. Let's load them:
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Load burned index
    import rioxarray
    burned = rioxarray.open_rasterio('burned.tif')

    # Load assets polygons
    import geopandas as gpd
    assets_raw = gpd.read_file('assets.gpkg')
    return (assets_raw, burned, gpd, rioxarray)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Align datasets

    Before we continue, let's check if the two datasets are in the same CRS:
    """)
    return


@app.cell
def _(assets, burned):
    print(assets.crs)
    print(burned.rio.crs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two datasets are in different CRS. Let's reproject the assets to the same CRS as the burned index raster:
    """)
    return


@app.cell
def _(assets_raw, burned):
    assets = assets_raw.to_crs(burned.rio.crs)
    return (assets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rasterize the vector data

    One way to define the zones is to create a grid space with the same extent and resolution as the burned index
    raster, and with the numerical values in the grid representing the type of infrastructure, i.e., the zones.
    This can be done by rasterizing the vector data `assets` to the grid space of `burned`.

    Let's first take two elements from `assets`, the geometry column, and the code of the region.
    """)
    return


@app.cell
def _(assets):
    geom = assets[['geometry', 'code']].values.tolist()
    geom
    return (geom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The raster image `burned` is a 3D image with a "band" dimension. To create the grid space, we only need the two
    spatial dimensions. We can use `.squeeze()` to drop the band dimension:
    """)
    return


@app.cell
def _(burned):
    print(burned.shape)
    burned_squeeze = burned.squeeze()
    print(burned_squeeze.shape)
    return (burned_squeeze,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can use `features.rasterize` from `rasterio` to rasterize the vector data `assets` to the grid space
    of `burned`:
    """)
    return


@app.cell
def _(burned, burned_squeeze, geom):
    from rasterio import features
    assets_rasterized = features.rasterize(
        geom,
        out_shape=burned_squeeze.shape,
        transform=burned.rio.transform()
    )
    assets_rasterized
    return (assets_rasterized, features)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perform zonal statistics

    The rasterized zones `assets_rasterized` is a `numpy` array. The Python package `xrspatial`, which is the one
    we will use for zonal statistics, accepts `xarray.DataArray`. We need to first convert `assets_rasterized`.
    We can use `burned_squeeze` as a template:
    """)
    return


@app.cell
def _(assets_rasterized, burned_squeeze):
    assets_rasterized_xarr = burned_squeeze.copy()
    assets_rasterized_xarr.data = assets_rasterized
    assets_rasterized_xarr.plot()
    return (assets_rasterized_xarr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E10/zones_rasterized_xarray.png)

    Then we can calculate the zonal statistics using the `zonal_stats` function:
    """)
    return


@app.cell
def _(assets_rasterized_xarr, burned_squeeze):
    from xrspatial import zonal_stats
    stats = zonal_stats(assets_rasterized_xarr, burned_squeeze)
    stats
    return (stats, zonal_stats)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results provide statistics for three zones: `1` represents infrastructure regions, `2` represents built-up
    regions, and `0` represents the rest of the area.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - Zones can be extracted by attribute columns of a vector dataset
    - Zones can be rasterized using `rasterio.features.rasterize`
    - Calculate zonal statistics with `xrspatial.zonal_stats` over the rasterized zones.
    """)
    return


if __name__ == "__main__":
    app.run()

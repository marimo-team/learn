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
#     "pystac",
# ]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Raster Calculations in Python
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions

    - How do I perform calculations on rasters and extract pixel values for defined locations?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objectives

    - Carry out operations with two rasters using Python's built-in math operators.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    We often want to combine values of and perform calculations on rasters to create a new output raster. This episode
    covers how to perform basic math operations using raster datasets. It also illustrates how to match rasters with
    different resolutions so that they can be used in the same calculation. As an example, we will calculate
    [a binary classification mask](https://custom-scripts.sentinel-hub.com/sentinel-2/burned_area_ms/) to identify burned
    area over a satellite scene.

    The classification mask requires the following of [the Sentinel-2 bands](https://gisgeography.com/sentinel-2-bands-combinations/)
    (and derived indices):

    * [Normalized difference vegetation index (NDVI)](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index),
      derived from the **near-infrared (NIR)** and **red** bands:

    $$ NDVI = \frac{NIR - red}{NIR + red} $$

    * [Normalized difference water index (NDWI)](https://en.wikipedia.org/wiki/Normalized_difference_water_index), derived
      from the **green** and **NIR** bands:

    $$ NDWI = \frac{green - NIR}{green + NIR} $$

    * A custom index derived from two of the **short-wave infrared (SWIR)** bands (with wavelenght ~1600 nm and ~2200 nm,
      respectively):

    $$ INDEX = \frac{SWIR_{16} - SWIR_{22}}{SWIR_{16} + SWIR_{22}}$$

    * The **blue**, **NIR**, and **SWIR** (1600 nm) bands.

    In the following, we start by computing the NDVI.

    ### Load and crop the Data

    For this episode, we will use one of the Sentinel-2 scenes that we have already employed in the previous episodes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Introduce the Data**

    We will use satellite images from the search that we have carried out in [the episode: "Access satellite imagery
    using Python"](05-access-data.md). Briefly, we have searched for Sentinel-2 scenes of Rhodes from July 1st to
    August 31st 2023 that have less than 1% cloud coverage. The search resulted in 11 scenes. We focus here on the
    most recent scene (August 27th), since that would show the situation after the wildfire, and use this as an example
    to demonstrate raster calculations.

    For your convenience, we have included the scene of interest among the datasets that you have already downloaded
    when following [the setup instructions](../learners/setup.md). You should, however, be able to download the
    satellite images "on-the-fly" using the JSON metadata file that was created in
    [the previous episode](05-access-data.md) (the file `rhodes_sentinel-2.json`).

    If you choose to work with the provided data (which is advised in case you are working offline or have a
    slow/unstable network connection) you can skip the remaining part of the block.

    If you want instead to experiment with downloading the data on-the-fly, you need to load the file
    `rhodes_sentinel-2.json`, which contains information on where and how to access the target satellite images
    from the remote repository.
    """), kind="info")
    return


@app.cell
def _():
    import pystac
    items = pystac.ItemCollection.from_file("rhodes_sentinel-2.json")
    return (items, pystac)


@app.cell
def _(items):
    item = items[0]
    print(item)
    return (item,)


@app.cell
def _(item):
    rhodes_red_href = item.assets["red"].href       # red band
    rhodes_green_href = item.assets["green"].href   # green band
    rhodes_blue_href = item.assets["blue"].href     # blue band
    rhodes_nir_href = item.assets["nir"].href       # near-infrared band
    rhodes_swir16_href = item.assets["swir16"].href # short-wave infrared (1600 nm) band
    rhodes_swir22_href = item.assets["swir22"].href # short-wave infrared (2200 nm) band
    rhodes_visual_href = item.assets["visual"].href # true-color image
    return (rhodes_blue_href, rhodes_green_href, rhodes_nir_href,
            rhodes_red_href, rhodes_swir16_href, rhodes_swir22_href,
            rhodes_visual_href)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import rioxarray
    red = rioxarray.open_rasterio("data/sentinel2/red.tif", masked=True)
    nir = rioxarray.open_rasterio("data/sentinel2/nir.tif", masked=True)
    return (nir, red, rioxarray)


@app.cell
def _(nir, red):
    import geopandas
    rhodes = geopandas.read_file('rhodes.gpkg')
    rhodes_reprojected = rhodes.to_crs(red.rio.crs)
    bbox = rhodes_reprojected.total_bounds

    # crop the rasters
    red_clip = red.rio.clip_box(*bbox)
    nir_clip = nir.rio.clip_box(*bbox)
    return (bbox, geopandas, nir_clip, red_clip, rhodes, rhodes_reprojected)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now plot the two rasters. Using `robust=True` color values are stretched between the 2nd and 98th
    percentiles of the data, which results in clearer distinctions between high and low reflectances:
    """)
    return


@app.cell
def _(red_clip):
    red_clip.plot(robust=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E09/red-band.png)
    """)
    return


@app.cell
def _(nir_clip):
    nir_clip.plot(robust=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E09/NIR-band.png)

    The burned area is immediately evident as a dark spot in the NIR wavelength, due to the lack of reflection from
    the vegetation in the scorched area.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raster Math

    We can perform raster calculations by subtracting (or adding, multiplying, etc.) two rasters. In the geospatial
    world, we call this "raster math", and typically it refers to operations on rasters that have the same width and
    height (including `nodata` pixels). We can check the shapes of the two rasters in the following way:
    """)
    return


@app.cell
def _(nir_clip, red_clip):
    print(red_clip.shape, nir_clip.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The shapes of the two rasters match (and, not shown, the coordinates and the CRSs match too).

    Let's now compute the NDVI as a new raster using the formula presented above. We'll use `DataArray` objects so
    that we can easily plot our result and keep track of the metadata.
    """)
    return


@app.cell
def _(nir_clip, red_clip):
    ndvi = (nir_clip - red_clip) / (nir_clip + red_clip)
    print(ndvi)
    return (ndvi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now plot the output NDVI:
    """)
    return


@app.cell
def _(ndvi):
    ndvi.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E09/NDVI-map.png)

    Notice that the range of values for the output NDVI is between -1 and 1. Does this make sense for the selected
    region?

    Maps are great, but it can also be informative to plot histograms of values to better understand the
    distribution. We can accomplish this using a built-in xarray method we have already been using: `plot.hist()`
    """)
    return


@app.cell
def _(ndvi):
    ndvi.plot.hist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E09/NDVI-hist.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _challenge_text = mo.md(r"""
    ### Challenge: NDWI and custom index to detect burned areas

    Calculate the other two indices required to compute the burned area classification mask, specifically:

    * The [normalized difference water index (NDWI)](https://en.wikipedia.org/wiki/Normalized_difference_water_index),
      derived from the **green** and **NIR** bands (file "green.tif" and "nir.tif", respectively):

    $$ NDWI = \frac{green - NIR}{green + NIR} $$

    * A custom index derived from the 1600 nm and the 2200 nm **short-wave infrared (SWIR)** bands ("swir16.tif"
      and "swir22.tif", respectively):

    $$ INDEX = \frac{SWIR_{16} - SWIR_{22}}{SWIR_{16} + SWIR_{22}}$$

    What challenge do you foresee in combining the data from the two indices?
    """)
    _solution = mo.accordion({"Solution": mo.md(r"""
    See the solution cells below for the full implementation. The key challenge is that the SWIR bands (and thus the
    derived custom index) have **lower resolution** (20 m) compared to the NIR and green bands (10 m), so the
    resolutions must be matched before combining the indices.
    """)})
    mo.vstack([_challenge_text, _solution])
    return


@app.cell
def _(rioxarray, bbox):
    def get_band_and_clip(band_path, bbox):
        band = rioxarray.open_rasterio(band_path, masked=True)
        return band.rio.clip_box(*bbox)
    return (get_band_and_clip,)


@app.cell
def _(get_band_and_clip, bbox):
    data_path = 'data/sentinel2'
    green_clip = get_band_and_clip(f'{data_path}/green.tif', bbox)
    swir16_clip = get_band_and_clip(f'{data_path}/swir16.tif', bbox)
    swir22_clip = get_band_and_clip(f'{data_path}/swir22.tif', bbox)
    return (data_path, green_clip, swir16_clip, swir22_clip)


@app.cell
def _(green_clip, nir_clip, swir16_clip, swir22_clip):
    ndwi = (green_clip - nir_clip) / (green_clip + nir_clip)
    index = (swir16_clip - swir22_clip) / (swir16_clip + swir22_clip)
    return (index, ndwi)


@app.cell
def _(ndwi):
    ndwi.plot(robust=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E09/NDWI.png)
    """)
    return


@app.cell
def _(index):
    index.plot(robust=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E09/custom-index.png)

    The challenge in combining the different indices is that the SWIR bands (and thus the derived custom index) have
    lower resolution:
    """)
    return


@app.cell
def _(index, ndvi, ndwi):
    ndvi.rio.resolution(), ndwi.rio.resolution(), index.rio.resolution()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to combine data from the computed indices, we use the `reproject_match` method, which reprojects, clips
    and matches the resolution of a raster using another raster as a template. We use the `ndvi` raster as a
    template, and match `index` and `swir16_clip` to its resolution and extent:
    """)
    return


@app.cell
def _(index, ndvi, swir16_clip):
    index_match = index.rio.reproject_match(ndvi)
    swir16_match = swir16_clip.rio.reproject_match(ndvi)
    return (index_match, swir16_match)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we also load the blue band data and clip it to the area of interest:
    """)
    return


@app.cell
def _(bbox, data_path, get_band_and_clip):
    blue_clip = get_band_and_clip(f'{data_path}/blue.tif', bbox)
    return (blue_clip,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now go ahead and compute the binary classification mask for burned areas. Note that we need to convert
    the unit of the Sentinel-2 bands
    [from digital numbers to reflectance](https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#units)
    (this is achieved by dividing by 10,000):
    """)
    return


@app.cell
def _(blue_clip, index_match, ndvi, ndwi, nir_clip, swir16_match):
    burned_full = (
        (ndvi <= 0.3) &
        (ndwi <= 0.1) &
        ((index_match + nir_clip / 10_000) <= 0.1) &
        ((blue_clip / 10_000) <= 0.1) &
        ((swir16_match / 10_000) >= 0.1)
    )
    return (burned_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The classification mask has a single element along the "band" axis, we can drop this dimension in the following
    way:
    """)
    return


@app.cell
def _(burned_full):
    burned = burned_full.squeeze()
    return (burned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now fetch and visualize the true color image of Rhodes, after coloring the pixels identified as burned
    area in red:
    """)
    return


@app.cell
def _(bbox, data_path, rioxarray):
    visual = rioxarray.open_rasterio(f'{data_path}/visual.tif')
    visual_clip = visual.rio.clip_box(*bbox)
    return (visual, visual_clip)


@app.cell
def _(burned, visual_clip):
    # set red channel to max (255), green and blue channels to min (0).
    visual_clip[0] = visual_clip[0].where(~burned, 255)
    visual_clip[1:3] = visual_clip[1:3].where(~burned, 0)
    return


@app.cell
def _(visual_clip):
    visual_clip.plot.imshow()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E09/visual-burned-index.png)

    We can save the burned classification mask to disk after converting booleans to integers:
    """)
    return


@app.cell
def _(burned):
    burned.fillna(0).astype('int8').rio.write_nodata(-1).rio.to_raster('burned.tif')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - Python's built-in math operators are fast and simple options for raster math.
    """)
    return


if __name__ == "__main__":
    app.run()

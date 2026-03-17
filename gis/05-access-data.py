# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.9.0",
#     "pystac-client",
#     "pystac",
#     "rioxarray",
#     "shapely",
# ]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Access satellite imagery using Python
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions

    - Where can I find open-access satellite data?
    - How do I search for satellite imagery with the STAC API?
    - How do I fetch remote raster datasets using Python?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objectives

    - Search public STAC repositories of satellite imagery using Python.
    - Inspect search result's metadata.
    - Download (a subset of) the assets available for a satellite scene.
    - Open satellite imagery as raster data and save it to disk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Considerations for the position of this episode in the workshop

    *When this workshop is taught to learners with limited prior knowledge of Python, it might be better to place this episode after episode 11 and before episode 12. This episode contains an introduction to working with APIs and dictionaries, which can be perceived as challenging by some learners. Another consideration for placing this episode later in the workshop is when it is taught to learners with prior GIS knowledge who want to perform GIS-like operations with data they have already collected or for learners interested in working with raster data but less interested in satellite images.*

    ## Introduction

    A number of satellites take snapshots of the Earth's surface from space. The images recorded by these remote sensors
    represent a very precious data source for any activity that involves monitoring changes on Earth. Satellite imagery is
    typically provided in the form of geospatial raster data, with the measurements in each grid cell ("pixel") being
    associated to accurate geographic coordinate information.

    In this episode we will explore how to access open satellite data using Python. In particular, we will
    consider [the Sentinel-2 data collection that is hosted on Amazon Web Services (AWS)](https://registry.opendata.aws/sentinel-2-l2a-cogs).
    This dataset consists of multi-band optical images acquired by the constellation of two satellites from
    [the Sentinel-2 mission](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-data/sentinel-2) and it is continuously updated with
    new images.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Search for satellite imagery

    ### The SpatioTemporal Asset Catalog (STAC) specification

    Current sensor resolutions and satellite revisit periods are such that terabytes of data products are added daily to the
    corresponding collections. Such datasets cannot be made accessible to users via full-catalog download. Therefore, space
    agencies and other data providers often offer access to their data catalogs through interactive Graphical User Interfaces
    (GUIs), see for instance the [Copernicus Browser](https://browser.dataspace.copernicus.eu) for the Sentinel missions.
    Accessing data via a GUI is a nice way to explore a catalog and get familiar with its content, but it represents a heavy
    and error-prone task that should be avoided if carried out systematically to retrieve data.

    A service that offers programmatic access to the data enables users to reach the desired data in a more reliable,
    scalable and reproducible manner. An important element in the software interface exposed to the users, which is generally
    called the Application Programming Interface (API), is the use of standards. Standards, in fact, can significantly
    facilitate the reusability of tools and scripts across datasets and applications.

    The SpatioTemporal Asset Catalog (STAC) specification is an emerging standard for describing geospatial data. By
    organizing metadata in a form that adheres to the STAC specifications, data providers make it possible for users to
    access data from different missions, instruments and collections using the same set of tools.

    ![Views of the STAC browser](/public/E05/STAC-browser.jpg)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **More Resources on STAC**

    - [STAC specification](https://github.com/radiantearth/stac-spec#readme)
    - [Tools based on STAC](https://stacindex.org/ecosystem)
    - [STAC catalogs](https://stacindex.org/catalogs)
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Search a STAC catalog

    The [STAC browser](https://radiantearth.github.io/stac-browser/#/) is a good starting point to discover available
    datasets, as it provides an up-to-date list of existing STAC catalogs. From the list, let's click on the
    "Earth Search" catalog, i.e. the access point to search the archive of Sentinel-2 images hosted on AWS.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _challenge_text = mo.md(r"""
    ### Challenge: Discover a STAC catalog

    Let's take a moment to explore the Earth Search STAC catalog, which is the catalog indexing the Sentinel-2 collection
    that is hosted on AWS. We can interactively browse this catalog using the STAC browser at
    [this link](https://radiantearth.github.io/stac-browser/#/external/earth-search.aws.element84.com/v1).

    1. Open the link in your web browser. Which (sub-)catalogs are available?
    2. Open the Sentinel-2 Level 2A collection, and select one item from the list. Each item corresponds to a satellite
    "scene", i.e. a portion of the footage recorded by the satellite at a given time. Have a look at the metadata fields
    and the list of assets. What kind of data do the assets represent?
    """)
    _solution = mo.accordion({"Solution": mo.md(r"""
    ![Views of the Earth Search STAC endpoint](/public/E05/STAC-browser-exercise.jpg)

    1. 8 sub-catalogs are available. In the STAC nomenclature, these are actually "collections", i.e. catalogs with
    additional information about the elements they list: spatial and temporal extents, license, providers, etc.
    Among the available collections, we have Landsat Collection 2, Level-2 and Sentinel-2 Level 2A (see left screenshot in
    the figure above).
    2. When you select the Sentinel-2 Level 2A collection, and randomly choose one of the items from the list, you
    should find yourself on a page similar to the right screenshot in the figure above. On the left side you will find
    a list of the available assets: overview images (thumbnail and true color images), metadata files and the "real"
    satellite images, one for each band captured by the Multispectral Instrument on board Sentinel-2.
    """)})
    mo.vstack([_challenge_text, _solution])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When opening a catalog with the STAC browser, you can access the API URL by clicking on the "Source" button on the top
    right of the page. By using this URL, you have access to the catalog content and, if supported by the catalog, to the
    functionality of searching its items. For the Earth Search STAC catalog the API URL is:
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    api_url = "https://earth-search.aws.element84.com/v1"
    return (api_url,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can query a STAC API endpoint from Python using the [`pystac_client` library](https://pystac-client.readthedocs.io).
    To do so we will first import `Client` from `pystac_client` and use the
    [method `open` from the Client object](https://pystac-client.readthedocs.io/en/stable/quickstart.html):
    """)
    return


@app.cell
def _(api_url):
    from pystac_client import Client

    client = Client.open(api_url)
    return (Client, client)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this episode we will focus at scenes belonging to the `sentinel-2-l2a` collection.
    This dataset is useful for our case and includes Sentinel-2 data products pre-processed at level 2A
    (bottom-of-atmosphere reflectance).

    In order to see which collections are available in the provided `api_url` the
    [`get_collections`](https://pystac-client.readthedocs.io/en/stable/api.html#pystac_client.Client.get_collections)
    method can be used on the Client object.
    """)
    return


@app.cell
def _(client):
    collections = client.get_collections()
    return (collections,)


@app.cell
def _(collections):
    for _collection in collections:
        print(_collection)
    return


@app.cell
def _():
    collection_sentinel_2_l2a = "sentinel-2-l2a"
    return (collection_sentinel_2_l2a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data in this collection is stored in the Cloud Optimized GeoTIFF (COG) format and as JPEG2000 images. In this
    episode we will focus at COGs, as these offer useful functionalities for our purpose.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Cloud Optimized GeoTIFFs**

    Cloud Optimized GeoTIFFs (COGs) are regular GeoTIFF files with some additional features that make them ideal to be
    employed in the context of cloud computing and other web-based services. This format builds on the widely-employed
    GeoTIFF format, already introduced in [Episode 1: Introduction to Raster Data](01-intro-raster-data.md).
    In essence, COGs are regular GeoTIFF files with a special internal structure. One of the features of COGs is that data
    is organized in "blocks" that can be accessed remotely via independent HTTP requests. Data users can thus access the
    only blocks of a GeoTIFF that are relevant for their analysis, without having to download the full file. In addition,
    COGs typically include multiple lower-resolution versions of the original image, called "overviews", which can also be
    accessed independently. By providing this "pyramidal" structure, users that are not interested in the details provided
    by a high-resolution raster can directly access the lower-resolution versions of the same image, significantly saving
    on the downloading time. More information on the COG format can be found [here](https://www.cogeo.org).
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to get data for a specific location you can add longitude latitude coordinates (World Geodetic System 1984
    EPSG:4326) in your request. In order to do so we are using the `shapely` library to define a geometrical point.
    Below we have included a point on the island of Rhodes, which is the location of interest for our case study
    (i.e. Longitude: 27.95 | Latitude 36.20).
    """)
    return


@app.cell
def _():
    from shapely.geometry import Point
    point = Point(27.95, 36.20)  # Coordinates of a point on Rhodes
    return (Point, point)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: at this stage, we are only dealing with metadata, so no image is going to be downloaded yet. But even metadata
    can be quite bulky if a large number of scenes match our search! For this reason, we limit the search by the
    intersection of the point (by setting the parameter `intersects`) and assign the collection (by setting the parameter
    `collections`). More information about the possible parameters to be set can be found in the `pystac_client`
    documentation for the
    [Client's `search` method](https://pystac-client.readthedocs.io/en/stable/api.html#pystac_client.Client.search).

    We now set up our search of satellite images in the following way:
    """)
    return


@app.cell
def _(client, collection_sentinel_2_l2a, point):
    search = client.search(
        collections=[collection_sentinel_2_l2a],
        intersects=point,
    )
    return (search,)


@app.cell
def _(search):
    print(search.matched())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You will notice that more than 500 scenes match our search criteria. We are however interested in the period right
    before and after the wildfire of Rhodes. In the following exercise you will therefore have to add a time filter to
    our search criteria to narrow down our search for images of that period.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _challenge_text = mo.md(r"""
    ### Challenge: Search satellite scenes with a time filter

    Search for all the available Sentinel-2 scenes in the `sentinel-2-c1-l2a` collection that have been recorded between
    1st of July 2023 and 31st of August 2023 (few weeks before and after the time in which the wildfire took place).

    Hint: You can find the input argument and the required syntax in the documentation of `client.search` (which you can
    access from Python or [online](https://pystac-client.readthedocs.io/en/stable/api.html#pystac_client.Client.search))

    How many scenes are available?
    """)
    _solution = mo.accordion({"Solution": mo.md(r"""
    ```python
    search = client.search(
        collections=[collection_sentinel_2_l2a],
        intersects=point,
        datetime='2023-07-01/2023-08-31'
    )
    print(search.matched())
    ```

    This means that 12 scenes satisfy the search criteria.
    """)})
    mo.vstack([_challenge_text, _solution])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have added a time filter, we retrieve the metadata of the search results by calling the method
    `item_collection`:
    """)
    return


@app.cell
def _(client, collection_sentinel_2_l2a, point):
    search_filtered = client.search(
        collections=[collection_sentinel_2_l2a],
        intersects=point,
        datetime='2023-07-01/2023-08-31'
    )
    items = search_filtered.item_collection()
    return (items, search_filtered)


@app.cell
def _(items):
    print(len(items))
    return


@app.cell
def _(items):
    for _item in items:
        print(_item)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each of the items contains information about the scene geometry, its acquisition time, and other metadata that can be
    accessed as a dictionary from the `properties` attribute. To see which information Item objects can contain you can
    have a look at the [pystac documentation](https://pystac.readthedocs.io/en/latest/api/pystac.html#pystac.Item).

    Let us inspect the metadata associated with the first item of the search results. Let us first look at the collection
    date of the first item:
    """)
    return


@app.cell
def _(items):
    item = items[0]
    print(item.datetime)
    return (item,)


@app.cell
def _(item):
    print(item.geometry)
    print(item.properties)
    return


@app.cell
def _(item):
    print(item.properties['proj:code'])
    return


@app.cell(hide_code=True)
def _(mo):
    _challenge_text = mo.md(r"""
    ### Challenge: Search satellite scenes using metadata filters

    Let's add a filter on the cloud cover to select the only scenes with less than 1% cloud coverage. How many scenes do
    now match our search?

    Hint: generic metadata filters can be implemented via the `query` input argument of `client.search`, which requires
    the following syntax
    (see [docs](https://pystac-client.readthedocs.io/en/stable/usage.html#query-extension)):
    `query=['<property><operator><value>']`.
    """)
    _solution = mo.accordion({"Solution": mo.md(r"""
    ```python
    search = client.search(
        collections=[collection_sentinel_2_l2a],
        intersects=point,
        datetime='2023-07-01/2023-08-31',
        query=['eo:cloud_cover<1']
    )
    print(search.matched())
    ```

    Result: 11 scenes match the criteria.
    """)})
    mo.vstack([_challenge_text, _solution])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once we are happy with our search, we save the search results in a file:
    """)
    return


@app.cell
def _(client, collection_sentinel_2_l2a, point):
    search_final = client.search(
        collections=[collection_sentinel_2_l2a],
        intersects=point,
        datetime='2023-07-01/2023-08-31',
        query=['eo:cloud_cover<1']
    )
    items_final = search_final.item_collection()
    items_final.save_object("rhodes_sentinel-2.json")
    return (items_final, search_final)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This creates a file in GeoJSON format, which we can reuse here and in the next episodes. Note that this file contains
    the metadata of the files that meet our criteria. It does not include the data itself, only their metadata.

    To load the saved search results as a `ItemCollection` we can use
    [`pystac.ItemCollection.from_file()`](https://pystac.readthedocs.io/en/stable/api/item_collection.html). Through
    this, we are instructing Python to use the `from_file` method of the `ItemCollection` class from the `pystac` library
    to load data from the specified GeoJSON file:
    """)
    return


@app.cell
def _():
    import pystac
    items_loaded = pystac.ItemCollection.from_file("rhodes_sentinel-2.json")
    return (items_loaded, pystac)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The loaded item collection (`items_loaded`) is equivalent to the one returned earlier by `search.item_collection()`
    (`items`). You can thus perform the same actions on it: you can check the number of items (`len(items_loaded)`), you
    can loop over items (`for item in items_loaded: ...`), and you can access individual elements using their index
    (`items_loaded[0]`).

    ## Access the assets

    So far we have only discussed metadata - but how can one get to the actual images of a satellite scene (the "assets"
    in the STAC nomenclature)? These can be reached via links that are made available through the item's attribute
    `assets`. Let's focus on the last item in the collection: this is the oldest in time, and it thus corresponds to an
    image taken before the wildfires.
    """)
    return


@app.cell
def _(items_final):
    assets = items_final[-1].assets  # last item's asset dictionary
    print(assets.keys())
    return (assets,)


@app.cell
def _(assets):
    for _key, _asset in assets.items():
        print(f"{_key}: {_asset.title}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Among the other data files, assets include multiple raster data files (one per optical band, as acquired by the
    multi-spectral instrument), a thumbnail, a true-color image ("visual"), instrument metadata and scene-classification
    information ("SCL"). Let's get the URL link to the thumbnail, which gives us a glimpse of the Sentinel-2 scene:
    """)
    return


@app.cell
def _(assets):
    print(assets["thumbnail"].href)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This can be used to download the corresponding file:

    ![Overview of the true-color image ("thumbnail") before the wildfires on Rhodes](/public/E05/STAC-s2-preview-before.jpg)

    For comparison, we can check out the thumbnail of the most recent scene of the sequence considered (i.e. the first
    item in the item collection), which has been taken after the wildfires:
    """)
    return


@app.cell
def _(items_final):
    print(items_final[0].assets["thumbnail"].href)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![Overview of the true-color image ("thumbnail") after the wildfires on Rhodes](/public/E05/STAC-s2-preview-after.jpg)

    From the thumbnails alone we can already observe some dark spots on the island of Rhodes at the bottom right of the
    image!

    In order to open the high-resolution satellite images and investigate the scenes in more detail, we will be using the
    [`rioxarray` library](https://corteva.github.io/rioxarray). Note that this library can both work with local and
    remote raster data. At this moment, we will only quickly look at the functionality of this library. We will learn
    more about it in the next episode.

    Now let us focus on the red band by accessing the item `red` from the assets dictionary and get the Hypertext
    Reference (also known as URL) attribute using `.href` after the item selection.

    For now we are using [rioxarray to open the raster file](https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray-open-rasterio).
    """)
    return


@app.cell
def _(assets):
    import rioxarray
    red_href = assets["red"].href
    red = rioxarray.open_rasterio(red_href)
    print(red)
    return (red, red_href, rioxarray)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we want to save the data to our local machine using the
    [to_raster](https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_array.RasterArray.to_raster)
    method:
    """)
    return


@app.cell
def _(red):
    # save whole image to disk
    red.rio.to_raster("red.tif")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That might take a while, given there are over 10000 x 10000 = a hundred million pixels in the 10-meter NIR band.
    But we can take a smaller subset before downloading it. Because the raster is a COG, we can download just what we need!

    In order to do that, we are using rioxarray's
    [`clip_box`](https://corteva.github.io/rioxarray/stable/examples/clip_box.html) with which you can set a bounding
    box defining the area you want.
    """)
    return


@app.cell
def _(red):
    red_subset = red.rio.clip_box(
        minx=560900,
        miny=3995000,
        maxx=570900,
        maxy=4015000
    )
    return (red_subset,)


@app.cell
def _(red_subset):
    red_subset.rio.to_raster("red_subset.tif")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The difference is 241 Megabytes for the full image vs less than 10 Megabytes for the subset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _challenge_text = mo.md(r"""
    ### Challenge: Downloading Landsat 8 Assets

    In this exercise we put in practice all the skills we have learned in this episode to retrieve images from a different
    mission: [Landsat 8](https://www.usgs.gov/landsat-missions/landsat-8). In particular, we browse images from the
    [Harmonized Landsat Sentinel-2 (HLS) project](https://lpdaac.usgs.gov/products/hlsl30v002/), which provides images
    from NASA's Landsat 8 and ESA's Sentinel-2 that have been made consistent with each other. The HLS catalog is indexed
    in the NASA Common Metadata Repository (CMR) and it can be accessed from the STAC API endpoint at the following URL:
    `https://cmr.earthdata.nasa.gov/stac/LPCLOUD`.

    - Using `pystac_client`, search for all assets of the Landsat 8 collection (`HLSL30.v2.0`) from February to March
      2021, intersecting the point with longitude/latitude coordinates (-73.97, 40.78) deg.
    - Visualize an item's thumbnail (asset key `browse`).
    """)
    _solution = mo.accordion({"Solution": mo.md(r"""
    ```python
    # connect to the STAC endpoint
    cmr_api_url = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
    client = Client.open(cmr_api_url)

    # setup search
    search = client.search(
        collections=["HLSL30.v2.0"],
        intersects=Point(-73.97, 40.78),
        datetime="2021-02-01/2021-03-30",
    ) # nasa cmr cloud cover filtering is currently broken: https://github.com/nasa/cmr-stac/issues/239

    # retrieve search results
    items = search.item_collection()
    print(len(items))

    items_sorted = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])
    item = items_sorted[0]
    print(item)

    print(item.assets["browse"].href)
    ```

    ![Thumbnail of the Landsat-8 scene](/public/E05/STAC-l8-preview.jpg)
    """)})
    mo.vstack([_challenge_text, _solution])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Public catalogs, protected data**

    Publicly accessible catalogs and STAC endpoints do not necessarily imply publicly accessible data. Data providers,
    in fact, may limit data access to specific infrastructures and/or require authentication. For instance, the NASA CMR
    STAC endpoint considered in the last exercise offers publicly accessible metadata for the HLS collection, but most
    of the linked assets are available only for registered users (the thumbnail is publicly accessible).

    The authentication procedure for dataset with restricted access might differ depending on the data provider. For the
    NASA CMR, follow these steps in order to access data using Python:

    * Create a NASA Earthdata login account [here](https://urs.earthdata.nasa.gov);
    * Set up a netrc file with your credentials, e.g. by using
      [this script](https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse/EarthdataLoginSetup.py);
    * Define the following environment variables (see code cell below).
    """), kind="info")
    return


@app.cell
def _():
    import os
    os.environ["GDAL_HTTP_COOKIEFILE"] = "./cookies.txt"
    os.environ["GDAL_HTTP_COOKIEJAR"] = "./cookies.txt"
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - Accessing satellite images via the providers' API enables a more reliable and scalable data retrieval.
    - STAC catalogs can be browsed and searched using the same tools and scripts.
    - `rioxarray` allows you to open and download remote raster files.
    """)
    return


if __name__ == "__main__":
    app.run()

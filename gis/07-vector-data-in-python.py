# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.9.0",
#     "geopandas==1.1.3",
#     "shapely",
#     "pandas",
#     "matplotlib==3.10.8",
#     "folium>=0.12",
#     "mapclassify==2.10.0",
# ]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Vector data in Python
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions

    - How can I read, inspect, and process spatial objects, such as points, lines, and polygons?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objectives

    - Load spatial objects.
    - Select the spatial objects within a bounding box.
    - Perform a CRS conversion of spatial objects.
    - Select features of spatial objects.
    - Match objects in two datasets based on their spatial relationships.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    In the preceding episodes, we have prepared, selected and downloaded raster data from before and after the wildfire
    event in the summer of 2023 on the Greek island of Rhodes. To evaluate the impact of this wildfire on the vital
    infrastructure and built-up areas we are going to create a subset of vector data representing these assets. In this
    episode you will learn how to extract vector data with specific characteristics like the type of attributes or their
    locations. The dataset that we will generate in this episode can later on be confronted with scorched areas which
    we determine by analyzing the satellite images
    [Episode 9: Raster Calculations in Python](09-raster-calculations.md).

    We'll be examining vector datasets that represent the valuable assets of Rhodes. As mentioned in
    [Episode 2: Introduction to Vector Data](02-intro-vector-data.md), vector data uses points, lines, and polygons
    to depict specific features on the Earth's surface. These geographic elements can have one or more attributes,
    like 'name' and 'population' for a city. In this episode we'll be using two open data sources: the Database of
    Global Administrative Areas (GADM) dataset to generate a polygon for the island of Rhodes and Open Street Map
    data for the vital infrastructure and valuable assets.

    To handle the vector data in python we use the package [`geopandas`](https://geopandas.org/en/stable/). This
    package allows us to open, manipulate, and write vector dataset through python.

    ![](/public/E07/pandas_geopandas_relation.png)

    `geopandas` enhances the widely-used `pandas` library for data analysis by extending its functionality to
    geospatial applications. The primary `pandas` objects (`Series` and `DataFrame`) are extended to `geopandas`
    objects (`GeoSeries` and `GeoDataFrame`). This extension is achieved by incorporating geometric types,
    represented in Python using the `shapely` library, and by offering dedicated methods for spatial operations like
    `union`, `spatial joins` and `intersect`. In order to understand how geopandas works, it is good to provide a
    brief explanation of the relationship between `Series`, a `DataFrame`, `GeoSeries`, and a `GeoDataFrame`:

    - A `Series` is a one-dimensional array with an axis that can hold any data type (integers, strings,
      floating-point numbers, Python objects, etc.)
    - A `DataFrame` is a two-dimensional labeled data structure with columns that can potentially hold different
      types of data.
    - A `GeoSeries` is a `Series` object designed to store shapely geometry objects.
    - A `GeoDataFrame` is an extended `pandas.DataFrame` that includes a column with geometry objects, which is a
      `GeoSeries`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Introduce the Vector Data**

    In this episode, we will use the downloaded vector data from the `data` directory. Please refer to the
    [setup page](../learners/setup.md) on where to download the data. Note that we manipulated that data a little
    for the purposes of this workshop. The link to the original source can be found on the
    [setup page](../learners/setup.md).
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get the administration boundary of study area

    The first thing we want to do is to extract a polygon containing the boundary of the island of Rhodes from Greece.
    For this we will use the [GADM dataset](https://gadm.org/download_country.html) layer `ADM_ADM_3.gpkg` for Greece.
    For your convenience we saved a copy at: `data/gadm/ADM_ADM_3.gpkg`. We will use the `geopandas` package to load
    the file and use the `read_file` function
    (see [docs](https://geopandas.org/en/stable/docs/user_guide/io.html)). Note that geopandas is often abbreviated
    as gpd.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import geopandas as gpd
    gdf_greece = gpd.read_file('data/gadm/ADM_ADM_3.gpkg')
    return (gdf_greece, gpd)


@app.cell
def _(gdf_greece):
    gdf_greece
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data are read into the variable fields as a `GeoDataFrame`. This is an extended data format of
    `pandas.DataFrame`, with an extra column `geometry`. To explore the dataframe you can call this variable just like
    a `pandas dataframe` by using functions like `.shape`, `.head` and `.tail` etc.

    To visualize the polygons we can use the
    [`plot()`](https://geopandas.org/en/stable/docs/user_guide/mapping.html) function to the `GeoDataFrame` we have
    loaded `gdf_greece`:
    """)
    return


@app.cell
def _(gdf_greece):
    gdf_greece.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E07/greece_administration_areas.png)

    If you want to interactively explore your data you can use the
    [`.explore`](https://geopandas.org/en/stable/docs/user_guide/interactive_mapping.html) function in geopandas:
    """)
    return


@app.cell
def _(gdf_greece):
    gdf_greece.explore()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this interactive map you can easily zoom in and out and hover over the polygons to see which attributes, stored
    in the rows of your GeoDataFrame, are related to each polygon.

    Next, we'll focus on isolating the administrative area of Rhodes Island. Once you hover over the polygon of
    Rhodos (the relatively big island to the east) you will find out that the label `Rhodos` is stored in the
    `NAME_3` column of `gdf_greece`, where Rhodes Island is listed as `Rhodos`. Since our goal is to have a boundary
    of Rhodes, we'll now create a new variable that exclusively represents Rhodes Island.

    To select an item in our GeoDataFrame with a specific value is done the same way in which this is done in a
    pandas `DataFrame` using
    [`.loc`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html).
    """)
    return


@app.cell
def _(gdf_greece):
    gdf_rhodes = gdf_greece.loc[gdf_greece['NAME_3'] == 'Rhodos']
    return (gdf_rhodes,)


@app.cell
def _(gdf_rhodes):
    gdf_rhodes.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E07/rhodes_administration_areas.png)

    Now that we have the administrative area of Rhodes Island. We can use the `to_file()` function to save this file
    for future use.
    """)
    return


@app.cell
def _(gdf_rhodes):
    gdf_rhodes.to_file('rhodes.gpkg')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get the vital infrastructure and built-up areas

    ### Road data from Open Street Map (OSM)

    Now that we have the boundary of our study area, we will make use of this to select the main roads in our study
    area. We will make the following processing steps:

    1. Select roads of study area
    2. Select key infrastructure: 'primary', 'secondary', 'tertiary'
    3. Create a 100m buffer around the roads. This buffer will be regarded as the infrastructure region. (note that
       this buffer is arbitrary and can be changed afterwards if you want!)

    #### Step 1: Select roads of study area

    For this workshop, in particular to not have everyone downloading too much data, we created a subset of the
    [Openstreetmap](https://www.openstreetmap.org/) data we downloaded for Greece from
    [the Geofabrik](https://download.geofabrik.de/europe.html). This data comes in the form of a shapefile
    (see [episode 2](02-intro-vector-data.md)) from which we extracted all the roads for `Rhodes` and some
    surrounding islands. The data is stored in the osm folder as `osm_roads.gpkg`, but contains *all* the roads on
    the island (so also hiking paths, private roads etc.), whereas we in particular are interested in the key
    infrastructure which we consider to be roads classified as primary, secondary or tertiary roads.

    Let's load the file and plot it:
    """)
    return


@app.cell
def _(gpd):
    gdf_roads = gpd.read_file('data/osm/osm_roads.gpkg')
    return (gdf_roads,)


@app.cell
def _(gdf_roads):
    gdf_roads.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E07/greece_highways.png)

    As you may have noticed, loading and plotting `osm_roads.gpkg` takes a bit long. This is because the file contains
    all the roads of Rhodos and some surrounding islands as well. Since we are only interested in the roads on Rhodes
    Island, we will use the [`mask`](https://geopandas.org/en/stable/docs/user_guide/io.html) parameter of the
    `read_file()` function to load only the roads on Rhodes Island.

    Now let us overwrite the GeoDataframe `gdf_roads` using the mask with the GeoDataFrame `gdf_rhodes` we created
    above.
    """)
    return


@app.cell
def _(gdf_rhodes, gpd):
    gdf_roads_rhodes = gpd.read_file('data/osm/osm_roads.gpkg', mask=gdf_rhodes)
    return (gdf_roads_rhodes,)


@app.cell
def _(gdf_roads_rhodes):
    gdf_roads_rhodes.explore()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E07/rhodes_highways.png)

    #### Step 2: Select key infrastructure

    As you will find out while exploring the roads dataset, information about the type of roads is stored in the
    `fclass` column. To get an overview of the different values that are present in the column `fclass`, we can use
    the [`unique()`](https://pandas.pydata.org/docs/reference/api/pandas.unique.html) function from pandas:
    """)
    return


@app.cell
def _(gdf_roads_rhodes):
    gdf_roads_rhodes['fclass'].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It seems the variable `gdf_roads` contains all kind of hiking paths and footpaths as well. Since we are only
    interested in vital infrastructure, classified as "primary", "secondary" and "tertiary" roads, we need to make a
    subselection.

    Let us first create a list with the labels we want to select.
    """)
    return


@app.cell
def _():
    key_infra_labels = ['primary', 'secondary', 'tertiary']
    return (key_infra_labels,)


@app.cell
def _(gdf_roads_rhodes, key_infra_labels):
    key_infra = gdf_roads_rhodes[gdf_roads_rhodes['fclass'].isin(key_infra_labels)]
    return (key_infra,)


@app.cell
def _(key_infra):
    key_infra.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E07/rhodes_infra_highways.png)

    #### Step 3: Create a 100m buffer around the key infrastructure

    Now that we selected the key infrastructure, we want to create a 100m buffer around them. This buffer will be
    regarded as the infrastructure region.

    As you might have noticed, the numbers on the x and y axis of our plots represent Lon Lat coordinates, meaning
    that the data is not yet projected. The current data has a geographic coordinate system with measures in degrees
    but not meters. Creating a buffer of 100 meters is not possible. Therefore, in order to create a 100m buffer, we
    first need to project our data. In our case we decided to project the data as WGS 84 / UTM zone 31N, with EPSG
    code 32631 ([see chapter 03 for more information about the CRS and EPSG codes](/episodes/03-crs.md)).

    To project our data we use
    [`.to_crs`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_crs.html). We first
    define a variable with the EPSG value (in our case 32631), which we then use in the to_crs function.
    """)
    return


@app.cell
def _(key_infra):
    epsg_code = 32631
    key_infra_meters = key_infra.to_crs(epsg_code)
    return (epsg_code, key_infra_meters)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that our data is projected, we can create a buffer. For this we make use of
    [geopandas' `.buffer` function](https://geopandas.org/en/stable/docs/user_guide/geometric_manipulations.html#GeoSeries.buffer):
    """)
    return


@app.cell
def _(key_infra_meters):
    key_infra_meters_buffer = key_infra_meters.buffer(100)
    key_infra_meters_buffer
    return (key_infra_meters_buffer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the type of the `key_infra_meters_buffer` is a `GeoSeries` and not a `GeoDataFrame`. This is because
    the `buffer()` function returns a `GeoSeries` object. You can check that by calling the type of the variable.
    """)
    return


@app.cell
def _(key_infra_meters_buffer):
    type(key_infra_meters_buffer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have a buffer, we can convert it back to the geographic coordinate system to keep the data consistent.
    Note that we are now using the crs information from the `key_infra`, instead of using the EPSG code directly
    (EPSG:4326):
    """)
    return


@app.cell
def _(key_infra, key_infra_meters_buffer):
    key_infra_buffer = key_infra_meters_buffer.to_crs(key_infra.crs)
    key_infra_buffer
    return (key_infra_buffer,)


@app.cell
def _(key_infra):
    print(key_infra.crs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reprojecting and buffering our data is something that we are going to do multiple times during this episode. To
    avoid having to call the same functions multiple times it would make sense to create a function. Therefore, let us
    create a function in which we can add the buffer as a variable.
    """)
    return


@app.cell
def _():
    def buffer_crs(gdf, size, meter_crs=32631, target_crs=4326):
        return gdf.to_crs(meter_crs).buffer(size).to_crs(target_crs)
    return (buffer_crs,)


@app.cell
def _(buffer_crs, key_infra):
    key_infra_buffer_200 = buffer_crs(key_infra, 200)
    return (key_infra_buffer_200,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get built-up regions from Open Street Map (OSM)

    Now that we have a buffered dataset for the key infrastructure of Rhodes, our next step is to create a dataset
    with all the built-up areas. To do so we will use the land use data from OSM, which we prepared for you in the
    file `data/osm_landuse.gpkg`. This file includes the land use data for the entire Greece. We assume the built-up
    regions to be the union of three types of land use: "commercial", "industrial", and "residential".

    Note that for the simplicity of this course, we limit the built-up regions to these three types of land use. In
    reality, the built-up regions can be more complex and there is definitely more high quality data available (e.g.
    from local government).

    Now it will be up to you to create a dataset with valuable assets. You should be able to complete this task by
    yourself with the knowledge you have gained from the previous steps and links to the documentation we provided.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _challenge_text = mo.md(r"""
    ### Challenge: Get the built-up regions

    Create a `builtup_buffer` from the file `data/osm/osm_landuse.gpkg` by the following steps:

    1. Load the land use data from `data/osm/osm_landuse.gpkg` and mask it with the administrative boundary of
       Rhodes Island (`gdf_rhodes`).
    2. Select the land use data for "commercial", "industrial", and "residential".
    3. Create a 10m buffer around the land use data.
    4. Visualize the results.

    After completing the exercise, answer the following questions:

    1. How many unique land use types are there in `osm_landuse.gpkg`?
    2. After selecting the three types of land use, how many entries (rows) are there in the results?

    Hints:

    - `data/osm_landuse.gpkg` contains the land use data for the entire Greece. Use the administrative boundary of
      Rhodes Island (`gdf_rhodes`) to select the land use data for Rhodes Island.
    - The land use attribute is stored in the `fclass` column.
    - Reuse `buffer_crs` function to create the buffer.
    """)
    _solution = mo.accordion({"Solution": mo.md(r"""
    ```python
    # Read data with a mask of Rhodes
    gdf_landuse = gpd.read_file('./data/osm/osm_landuse.gpkg', mask=gdf_rhodes)

    # Find number of unique landuse types
    print(len(gdf_landuse['fclass'].unique()))

    # Extract built-up regions
    builtup_labels = ['commercial', 'industrial', 'residential']
    builtup = gdf_landuse[gdf_landuse['fclass'].isin(builtup_labels)]

    # Create 10m buffer around the built-up regions
    builtup_buffer = buffer_crs(builtup, 10)

    # Get the number of entries
    print(len(builtup_buffer))

    # Visualize the buffer
    builtup_buffer.plot()
    ```

    Results: 19 unique land use types; 1349 entries after selection.

    ![](/public/E07/rhodes_builtup_buffer.png)
    """)})
    mo.vstack([_challenge_text, _solution])
    return


@app.cell
def _(buffer_crs, gdf_rhodes, gpd):
    gdf_landuse = gpd.read_file('./data/osm/osm_landuse.gpkg', mask=gdf_rhodes)
    builtup_labels = ['commercial', 'industrial', 'residential']
    builtup = gdf_landuse[gdf_landuse['fclass'].isin(builtup_labels)]
    builtup_buffer = buffer_crs(builtup, 10)
    builtup_buffer.plot()
    return (builtup, builtup_buffer, builtup_labels, gdf_landuse)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Merge the infrastructure regions and built-up regions

    Now that we have the infrastructure regions and built-up regions, we can merge them into a single region. We would
    like to keep track of the type after merging, so we will add two new columns: `type` and `code` by converting the
    `GeoSeries` to `GeoDataFrame`.

    First we convert the buffer around key infrastructure:
    """)
    return


@app.cell
def _(gpd, key_infra_buffer):
    data_infra = {'geometry': key_infra_buffer, 'type': 'infrastructure', 'code': 1}
    gdf_infra = gpd.GeoDataFrame(data_infra)
    return (data_infra, gdf_infra)


@app.cell
def _(builtup_buffer, gpd):
    data_builtup = {'geometry': builtup_buffer, 'type': 'builtup', 'code': 2}
    gdf_builtup = gpd.GeoDataFrame(data_builtup)
    return (data_builtup, gdf_builtup)


@app.cell
def _(gdf_builtup, gdf_infra):
    import pandas as pd
    gdf_assets = pd.concat([gdf_infra, gdf_builtup]).reset_index(drop=True)
    return (gdf_assets, pd)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In `gdf_assets`, we can distinguish the infrastructure regions and built-up regions by the `type` and `code`
    columns. We can plot the `gdf_assets` to visualize the merged regions. See the
    [geopandas documentation](https://geopandas.org/en/stable/docs/user_guide/mapping.html) on how to do this:
    """)
    return


@app.cell
def _(gdf_assets):
    gdf_assets.plot(column='type', legend=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](/public/E07/rhodes_assets.png)

    Finally, we can save the `gdf_assets` to a file for future use:
    """)
    return


@app.cell
def _(gdf_assets):
    gdf_assets.to_file('assets.gpkg')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - Load spatial objects into Python with `geopandas.read_file()` function.
    - Spatial objects can be plotted directly with `GeoDataFrame`'s `.plot()` method.
    - Convert CRS of spatial objects with `.to_crs()`. Note that this generates a `GeoSeries` object.
    - Create a buffer of spatial objects with `.buffer()`.
    - Merge spatial objects with `pd.concat()`.
    """)
    return


if __name__ == "__main__":
    app.run()

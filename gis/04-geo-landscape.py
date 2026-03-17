# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.9.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Geospatial Landscape
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions
    - What programs and applications are available for working with geospatial data?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objectives
    - Describe the difference between various approaches to geospatial computing, and their relative strengths and weaknesses.
    - Name some commonly used GIS applications.
    - Name some commonly used Python packages that can access and process spatial data.
    - Describe pros and cons for working with geospatial data using a command-line versus a graphical user interface.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Standalone Software Packages

    Most traditional GIS work is carried out in standalone applications that aim to provide
    end-to-end geospatial solutions. These applications are available under a wide range of
    licenses and price points. Some of the most common are listed below.

    ### Open-source software

    The [Open Source Geospatial Foundation (OSGEO)](https://www.osgeo.org/) supports several actively managed GIS platforms:

    * [QGIS](https://www.qgis.org/en/site/) is a professional GIS application that is
      built on top of and proud to be itself Free and Open Source Software (FOSS). QGIS is
      written in Python and C++, has a python console interface and allows one to develop plugins.
    * [GRASS GIS](https://grass.osgeo.org/), commonly referred to as GRASS
      (Geographic Resources Analysis Support System), is a FOSS-GIS software suite used for
      geospatial data management and analysis, image processing, graphics and maps
      production, spatial modeling, and visualization. GRASS GIS is currently used in
      academic and commercial settings around the world, as well as by many governmental
      agencies and environmental consulting companies. It is a founding member of the Open
      Source Geospatial Foundation (OSGeo). GRASS GIS can be installed along with and made
      accessible within QGIS 3.
    * [GDAL](https://www.gdal.org/) is a multiplatform
      set of tools for translating between geospatial data formats. It can also handle
      reprojection and a variety of geoprocessing tasks. GDAL is built in to many
      applications both FOSS and commercial, including GRASS and QGIS.
    * [SAGA-GIS](https://www.saga-gis.org/en/index.html), or System for Automated
      Geoscientific Analyses, is a FOSS-GIS application developed by a small team of
      researchers from the Dept. of Physical Geography, Göttingen, and the Dept. of
      Physical Geography, Hamburg. SAGA has been designed for an easy and effective
      implementation of spatial algorithms, offers a comprehensive, growing set of
      geoscientific methods, provides an easily approachable user interface with many
      visualisation options, and runs under Windows and Linux operating systems. Like
      GRASS GIS, it can also be installed and made accessible in QGIS3.
    * [PostGIS](https://postgis.net/) is a geospatial extension to the PostGreSQL
      relational database and is especially suited to work with large vector datasets.
    * [GeoDMS](https://geodms.nl/) is a powerful Open Source GIS software which allows for
      fast calculations and calculations with large datasets. Furthermore it allows for complex scenario analyses.

    ### Commercial software

    * [ESRI (Environmental Systems Research Institute)](https://www.esri.com/en-us/home)
      is an international supplier of geographic information system (GIS) software, web GIS
      and geodatabase management applications. ESRI provides several licenced platforms for
      performing GIS, including [ArcGIS](https://www.arcgis.com/home/index.html),
      [ArcGIS Online](https://www.esri.com/software/arcgis/arcgisonline), and
      [Portal for ArcGIS](https://server.arcgis.com/en/portal/) a standalone version of
      ArcGIS Online which you host locally. ESRI welcomes development on their platforms
      through their [DevLabs](https://developers.arcgis.com/). ArcGIS software can be
      installed using
      [Chef Cookbooks from Github](https://github.com/Esri/arcgis-cookbook). In addition, ESRI offers the [ArcPy Python library](https://pro.arcgis.com/en/pro-app/latest/arcpy/get-started/what-is-arcpy-.htm) as part of an ArcGIS Pro licence, allowing to translate operations in the ArcGIS Pro GUI to Python scripts.
    * Precisely produces [MapInfo Professional](https://www.precisely.com/product/precisely-mapinfo/mapinfo-pro), formerly developed by Pitney Bowes Software,
      which was one of the earliest desktop GIS programs on the market.
    * [Hexagon Geospatial Power Portfolio](https://www.hexagongeospatial.com/products/products)
      includes many geospatial tools including ERDAS Imagine, powerful software for remote sensing.
    * [Manifold](https://www.manifold.net/) is a desktop GIS that emphasizes speed through
      the use of parallel and GPU processing.

    ### Online + Cloud computing

    * [PANGEO](https://pangeo.io/) is a community organization dedicated to open and reproducible data science with python.
      They focus on the Pangeo software ecosystem for working with big
      data in the geosciences.
    * Google developed [Google Earth Engine](https://earthengine.google.com/) which
      combines a multi-petabyte catalog of satellite imagery and geospatial datasets with
      planetary-scale analysis capabilities and makes it available for scientists,
      researchers, and developers to detect changes, map trends, and quantify differences
      on the Earth's surface. [Earth Engine API](https://developers.google.com/earth-engine/)
      runs in both Python and JavaScript.
    * [ArcGIS Online](https://www.arcgis.com/features/features.html) provides access to
      thousands of maps and base layers.

    Private companies have released SDK platforms for large scale GIS analysis:

    * [Kepler.gl](https://kepler.gl/#/) is Uber's toolkit for handling large datasets (i.e. Uber's data archive).

    Publicly funded open-source platforms for large scale GIS analysis:

    * [PANGEO](https://pangeo.io/) for the Earth Sciences. This community organization also supports python libraries like xarray, iris, dask, jupyter, and many other packages.
    * [Sepal.io](https://sepal.io/) by [FAO Open Foris](https://www.openforis.org/) utilizing EOS satellite imagery and cloud resources for global forest monitoring.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GUI vs CLI

    The earliest computer systems operated without a graphical user interface (GUI),
    relying only on the command-line interface (CLI). Since mapping and spatial analysis
    are strongly visual tasks, GIS applications benefited greatly from the emergence of
    GUIs and quickly came to rely heavily on them. Most modern GIS applications have very
    complex GUIs, with all common tools and procedures accessed via buttons and menus.

    Benefits of using a GUI include:

    - Tools are all laid out in front of you
    - Complex commands are easy to build
    - Don't need to learn a coding language
    - Cartography and visualisation is more intuitive and flexible

    Downsides of using a GUI include:

    - Low reproducibility - you can record your actions and replay, but this is limited to the functionalities of the software
    - Batch-processing is possible, but limited to the functionalities of the software and hard to be integrated with other workflows
    - Limited ability to customise functions or write your own
    - Intimidating interface for new users - so many buttons!

    In scientific computing, the lack of reproducibility in point-and-click software has
    come to be viewed as a critical weakness. As such, scripted CLI-style workflows are
    becoming popular, which leads us to another approach to doing GIS — via a
    programming language. Therefore this is the approach we will be using throughout this workshop.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GIS in programming languages

    A number of powerful geospatial processing libraries exist for general-purpose
    programming languages like Java and C++. However, the learning curve for these
    languages is steep and the effort required is excessive for users who only need a
    subset of their functionality.

    Higher-level scripting languages like Python and R are considered easier to learn and use. Both
    now have their own packages that wrap up those geospatial processing libraries and make
    them easy to access and use safely. A key example is the Java Topology Suite (JTS),
    which is implemented in C++ as GEOS. GEOS is accessible in Python via the `shapely`
    package (and `geopandas`, which makes use of `shapely`) and in R via `sf`. R and Python
    also have interface packages for GDAL, and for specific GIS apps.

    This last point is a huge advantage for GIS-by-programming; these interface packages
    give you the ability to access functions unique to particular programs, but have your
    entire workflow recorded in a central document - a document that can be re-run at will.
    Below are lists of some of the key spatial packages for Python, which we will be using in the
    remainder of this workshop.

    * `geopandas` and `geocube` for working with vector data
    * `rasterio` and `rioxarray` for working with raster data

    These packages along with the `matplotlib` package are all we need for spatial data visualisation. Python also has many fundamental scientific packages that are relevant in the geospatial domain. Below is a list of particularly fundamental packages. `numpy`, `scipy`, and `scikit-image` are all excellent options for working with rasters, as arrays.

    An overview of these and other Python spatial packages can be [accessed here](https://medium.com/@chrieke/essential-geospatial-python-libraries-5d82fcc38731).

    As a programming language, Python can be a CLI tool. However, using
    Python together with an [Integrated Development Environment](https://www.codecademy.com/articles/what-is-an-ide) (IDE) application
    allows some GUI features to become part of your workflow. IDEs allow the best of both
    worlds. They provide a place to visually examine data and other software objects,
    interact with your file system, and draw plots and maps, but your activities are still
    command-driven: recordable and reproducible. There are several IDEs available for Python.
    [JupyterLab](https://jupyter.org/) is well-developed and the most widely used option for data science
    in Python. [VSCode](https://code.visualstudio.com/docs/python/python-tutorial) and [Spyder](https://www.spyder-ide.org/)
    are other popular options for data science.

    Traditional GIS apps are also moving back towards providing a scripting environment for
    users, further blurring the CLI/GUI divide. ESRI have adopted Python into their
    software by introducing [ArcPy](https://developers.arcgis.com/documentation/arcgis-add-ins-and-automation/arcpy/), and QGIS is both Python and R-friendly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GIS File Types

    There are a variety of file types that are used in GIS analysis. Depending on the program you choose to use some file
    types can be used while others are not readable. Below is a brief table describing some of the most common vector and
    raster file types.

    ### Vector

    | File Type | Extensions | Description |
    | :-------- | :--------- | :---------- |
    | Esri Shapefile | .SHP .DBF .SHX | The most common geospatial file type. This has become the industry standard. The three required files are: SHP is the feature geometry. SHX is the shape index position. DBF is the attribute data. |
    | GeoPackage | .gpkg | As an alternative to Shapefile, this open file format is gaining terrain and it consists of one file containing all necessary attribute information. |
    | Geographic JavaScript Object Notation (GeoJSON) | .GEOJSON .JSON | Used for web-based mapping and uses JavaScript Object Notation to store the coordinates as text. |
    | Google Keyhole Markup Language (KML) | .KML .KMZ | KML stands for Keyhole Markup Language. This GIS format is XML-based and is primarily used for Google Earth. |
    | GPX or GPS Exchange Format | .gpx | XML schema designed as a common GPS data format for software applications. This format is often used for tracking activities e.g. hiking, cycling, running etc. |
    | OpenStreetMap | .OSM | OSM files are the native file for OpenStreetMap which had become the largest crowdsourcing GIS data project in the world. These files are a collection of vector features from crowd-sourced contributions from the open community. |

    ### Raster

    | File Type | Extensions | Description |
    | :-------- | :--------- | :---------- |
    | ERDAS Imagine | .IMG | ERDAS Imagine IMG files is a proprietary file format developed by Hexagon Geospatial. IMG files are commonly used for raster data to store single and multiple bands of satellite data. Each raster layer as part of an IMG file contains information about its data values. For example, this includes projection, statistics, attributes, pyramids and whether or not it's a continuous or discrete type of raster. |
    | GeoTIFF | .TIF .TIFF .OVR | The GeoTIFF has become an industry image standard file for GIS and satellite remote sensing applications. GeoTIFFs may be accompanied by other files: TFW is the world file that is required to give your raster geolocation. XML optionally accompany GeoTIFFs and are your metadata. AUX auxiliary files store projections and other information. OVR pyramid files improves performance for raster display. |
    | Cloud Optimized GeoTIFF (COG) | .TIF .TIFF | Based on the GeoTIFF standard, COGs incorporate tiling and overviews to support HTTP range requests where users can query and load subsets of the image without having to transfer the entire file. |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points
    - Many software packages exist for working with geospatial data.
    - Command-line programs allow you to automate and reproduce your work.
    - JupyterLab provides a user-friendly interface for working with Python.
    """)
    return


if __name__ == "__main__":
    app.run()

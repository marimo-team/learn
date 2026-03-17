# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.9.0",
#     "rioxarray",
#     "rasterio",
#     "numpy",
#     "matplotlib",
#     "xarray",
#     "dask",
#     "pystac",
# ]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parallel raster computations using Dask
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions

    - How can I parallelize computations on rasters with Dask?
    - How can I determine if parallelization improves calculation speed?
    - What are good practices in applying parallelization to my raster calculations?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objectives

    - Profile the timing of the raster calculations.
    - Open raster data as a chunked array.
    - Recognize good practices in selecting proper chunk sizes.
    - Setup raster calculations that take advantage of parallelization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    Very often raster computations involve applying the same operation to different pieces of data. Think, for
    instance, to the "pixel"-wise sum of two raster datasets, where the same sum operation is applied to all the
    matching grid-cells of the two rasters. This class of tasks can benefit from chunking the input raster(s) into
    smaller pieces: operations on different pieces can be run in parallel using multiple computing units (e.g.,
    multi-core CPUs), thus potentially speeding up calculations. In addition, working on chunked data can lead to
    smaller memory footprints, since one may bypass the need to store the full dataset in memory by processing it
    chunk by chunk.

    In this episode, we will introduce the use of Dask in the context of raster calculations. Dask is a Python
    library for parallel and distributed computing. It provides a framework to work with different data structures,
    including chunked arrays (Dask Arrays). Dask is well integrated with (`rio`)`xarray`, which can use Dask arrays
    as underlying data structures.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Dask**

    This episode shows how Dask can be used to parallelize operations on local CPUs. However, the same library can
    be configured to run tasks on large compute clusters.

    More resources on Dask:

    - [Dask](https://dask.org) and [Dask Array](https://docs.dask.org/en/stable/array.html).
    - [Xarray with Dask](https://xarray.pydata.org/en/stable/user-guide/dask.html).
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is important to realize, however, that many details determine the extent to which using Dask's chunked arrays
    instead of regular Numpy arrays leads to faster calculations (and lower memory requirements). The actual
    operations to carry out, the size of the dataset, and parameters such as the chunks' shape and size, all affects
    the performance of our computations. Depending on the specifics of the calculations, serial calculations might
    actually turn out to be faster! Being able to profile the computational time is thus essential, and we will see
    how to do that in a Jupyter environment in the next section.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Introduce the Data**

    We will use satellite images from the search that we have carried out in [the episode: "Access satellite imagery
    using Python"](05-access-data.md). Briefly, we have searched for Sentinel-2 scenes of Rhodes from July 1st to
    August 31st 2023 that have less than 1% cloud coverage. The search resulted in 11 scenes. We focus here on the
    most recent scene (August 27th), since that would show the situation after the wildfire, and use this as an
    example to demonstrate parallel raster calculations.

    For your convenience, we have included the scene of interest among the datasets that you have already downloaded
    when following [the setup instructions](../learners/setup.md). You should, however, be able to download the
    satellite images "on-the-fly" using the JSON metadata file that was created in
    [the previous episode](05-access-data.md) (the file `rhodes_sentinel-2.json`).

    If you choose to work with the provided data (which is advised in case you are working offline or have a
    slow/unstable network connection) you can skip the remaining part of the block and continue with the following
    section: [Dask-powered rasters](#Dask-powered-rasters).

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
    rhodes_red_href = item.assets["red"].href  # red band
    rhodes_nir_href = item.assets["nir"].href  # near-infrared band
    return (rhodes_nir_href, rhodes_red_href)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dask-powered rasters

    ### Chunked arrays

    As we have mentioned, `rioxarray` supports the use of Dask's chunked arrays as underlying data structure. When
    opening a raster file with `open_rasterio` and providing the `chunks` argument, Dask arrays are employed
    instead of regular Numpy arrays. `chunks` describes the shape of the blocks which the data will be split in.
    As an example, we open the red band raster using a chunk shape of `(1, 4000, 4000)` (block size of `1` in the
    first dimension and of `4000` in the second and third dimensions):
    """)
    return


@app.cell
def _():
    import rioxarray
    red = rioxarray.open_rasterio("data/sentinel2/red.tif", chunks=(1, 4000, 4000))
    return (red, rioxarray)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![Xarray Dask-backed DataArray](/public/E11/xarray-with-dask.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _challenge_text = mo.md(r"""
    ### Challenge: Chunk sizes matter

    We have already seen how COGs are regular GeoTIFF files with a special internal structure. Another feature of
    COGs is that data is organized in "blocks" that can be accessed remotely via independent HTTP requests,
    enabling partial file readings. This is useful if you want to access only a portion of your raster file, but
    it also allows for efficient parallel reading. You can check the blocksize employed in a COG file with the
    following code snippet:

    ```python
    import rasterio
    with rasterio.open("/path/or/URL/to/file.tif") as r:
        if r.is_tiled:
            print(f"Chunk size: {r.block_shapes}")
    ```

    In order to optimally access COGs it is best to align the blocksize of the file with the chunks employed when
    loading the file. Which other elements do you think should be considered when choosing the chunk size? What do
    you think are suitable chunk sizes for the red band raster?
    """)
    _solution = mo.accordion({"Solution": mo.md(r"""
    See the solution cells below. Ideal chunk size values for this raster are multiples of 1024. An element to
    consider is the number of resulting chunks and their size. While the optimal chunk size strongly depends on
    the specific application, chunks should in general not be too big nor too small (i.e. too many). As a rule of
    thumb, chunk sizes of 100 MB typically work well with Dask. Also, the shape might be relevant, depending on
    the application!
    """)})
    mo.vstack([_challenge_text, _solution])
    return


@app.cell
def _():
    import rasterio
    with rasterio.open("data/sentinel2/red.tif") as r:
        if r.is_tiled:
            print(f"Chunk size: {r.block_shapes}")
    return (rasterio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we might select a chunks shape of `(1, 6144, 6144)`, which leads to chunks 72 MB large:
    ((1 x 6144 x 6144) x 2 bytes / 2^20 = 72 MB). We can also let rioxarray and Dask figure out appropriate
    chunk shapes by setting `chunks="auto"`, which leads to `(1, 8192, 8192)` chunks (128 MB).
    """)
    return


@app.cell
def _(rioxarray):
    red_chunked = rioxarray.open_rasterio("data/sentinel2/red.tif", chunks=(1, 6144, 6144))
    return (red_chunked,)


@app.cell
def _(rioxarray):
    red_auto = rioxarray.open_rasterio("data/sentinel2/red.tif", chunks="auto")
    return (red_auto,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parallel computations

    Operations performed on a `DataArray` that has been opened as a chunked Dask array are executed using Dask.
    Dask coordinates how the operations should be executed on the individual chunks of data, and runs these tasks
    in parallel as much as possible.

    Let us set up an example where we calculate the NDVI for a full Sentinel-2 tile, and try to estimate the
    performance gain by running the calculation in parallel on a multi-core CPU.

    To run the calculation serially, we open the relevant raster bands as we have learned in the previous episodes:
    """)
    return


@app.cell
def _(rioxarray):
    red_serial = rioxarray.open_rasterio('data/sentinel2/red.tif', masked=True)
    nir_serial = rioxarray.open_rasterio('data/sentinel2/nir.tif', masked=True)
    return (nir_serial, red_serial)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We then compute the NDVI and time it using Python's `time` module (in a Jupyter notebook you would use the
    `%%time` magic):
    """)
    return


@app.cell
def _(nir_serial, red_serial):
    import time
    _t0 = time.time()
    ndvi_serial = (nir_serial - red_serial) / (nir_serial + red_serial)
    _t1 = time.time()
    print(f"Serial wall time: {_t1 - _t0:.2f} s")
    return (ndvi_serial, time)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We note down the calculation's wall time (actual time to perform the task).

    Now we run the same task in parallel using Dask. To do so, we open the relevant rasters as chunked arrays.
    Setting `lock=False` tells `rioxarray` that the individual data chunks can be loaded simultaneously from the
    source by the Dask workers.
    """)
    return


@app.cell
def _(rioxarray):
    red_dask = rioxarray.open_rasterio('data/sentinel2/red.tif', masked=True, lock=False, chunks=(1, 6144, 6144))
    nir_dask = rioxarray.open_rasterio('data/sentinel2/nir.tif', masked=True, lock=False, chunks=(1, 6144, 6144))
    return (nir_dask, red_dask)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now compute the NDVI with Dask. Note how the same syntax as for the serial version is used:
    """)
    return


@app.cell
def _(nir_dask, red_dask, time):
    _t0 = time.time()
    ndvi_dask = (nir_dask - red_dask) / (nir_dask + red_dask)
    _t1 = time.time()
    print(f"Dask setup wall time: {_t1 - _t0:.4f} s")
    return (ndvi_dask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Did we just observe a ~1000x speed-up? Actually, no calculation has run yet. This is because operations
    performed on Dask arrays are executed "lazily", i.e. they are not immediately run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Dask graph**

    The sequence of operations to carry out is stored in a task graph, which can be visualized with:

    ```python
    import dask
    dask.visualize(ndvi_dask)
    ```

    ![Dask graph](/public/E11/dask-graph.png)

    The task graph gives Dask the complete "overview" of the calculation, thus enabling a better management of
    tasks and resources when dispatching calculations to be run in parallel.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most methods of `DataArray`'s run operations lazily when Dask arrays are employed. In order to trigger
    calculations, we can use either `.persist()` or `.compute()`. The former keeps data in the form of chunked
    Dask arrays, and it should thus be used to run intermediate steps that will be followed by additional
    calculations. The latter merges the chunks into a single Numpy array, and it should be used at the very end of
    a sequence of calculations. Both methods accept the same parameters. Here, we explicitly tell Dask to
    parallelize the required workload over 4 threads:
    """)
    return


@app.cell
def _(ndvi_dask, time):
    _t0 = time.time()
    ndvi_dask_2 = ndvi_dask.persist(scheduler="threads", num_workers=4)
    _t1 = time.time()
    print(f"Dask parallel wall time: {_t1 - _t0:.2f} s")
    return (ndvi_dask_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When running the task on a 4-core CPU laptop, we observe a roughly x3.6 speed-up when comparing to the
    analogous serial calculation. Once again, we stress that one does not always obtain similar performance gains
    by exploiting the Dask-based parallelization. Even if the algorithm employed is well suited for
    parallelization, Dask introduces some overhead time to manage the tasks in the Dask graph. This overhead,
    which is typically of the order of few milliseconds per task, can be larger than the parallelization gain.
    This is the typical situation with calculations with many small chunks.

    Finally, let's have a look at how Dask can be used to save raster files. When calling `.to_raster()`, we
    provide the additional argument `lock=threading.Lock()`. This is because the threads which are splitting the
    workload must "synchronise" when writing to the same file (they might otherwise overwrite each other's output).
    """)
    return


@app.cell
def _(ndvi_dask_2):
    from threading import Lock
    ndvi_dask_2.rio.to_raster('ndvi.tif', tiled=True, lock=Lock())
    return (Lock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that `.to_raster()` is among the methods that trigger immediate calculations (one can change this
    behaviour by specifying `compute=False`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Points

    - The `%%time` Jupyter magic command can be used to profile calculations.
    - Data 'chunks' are the unit of parallelization in raster calculations.
    - (`rio`)`xarray` can open raster files as chunked arrays.
    - The chunk shape and size can significantly affect the calculation performance.
    - Cloud-optimized GeoTIFFs have an internal structure that enables performant parallel read.
    """)
    return


if __name__ == "__main__":
    app.run()

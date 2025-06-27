# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "marimo",
#     "faker==37.1.0",
#     "scipy==1.13.1",
#     "numpy==2.0.2",
#     "numba==0.60.0",
#     "polars==1.26.0",
#     "matplotlib==3.9.4",
#     "statsmodels",
#     "pandas==2.2.3",
# ]
# ///

import marimo

__generated_with = "0.12.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Lazy Execution (a.k.a. the Lazy API)

        Author: [Deb Debnath](https://github.com/debajyotid2)
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import random
    import re
    import time
    from datetime import datetime, timedelta, timezone
    from typing import Generator
    import numba
    import numpy as np
    import polars as pl
    import pandas as pd
    import scipy.special as spl
    import scipy.stats as st
    import matplotlib.pyplot as plt
    from faker import Faker
    return (
        Faker,
        Generator,
        datetime,
        np,
        numba,
        pd,
        pl,
        plt,
        random,
        re,
        spl,
        st,
        time,
        timedelta,
        timezone,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We saw the benefits of lazy evaluation when we learned about the Expressions API in Polars. Lazy execution is further extended as a philosophy by the Lazy API. It offers significant performance enhancements over eager (immediate) execution of queries and is one of the reasons why Polars is faster at working with large (GB scale) datasets than other libraries. The lazy API optimizes the full query pipeline instead of executing individual queries optimally, unlike eager execution. Some of the advantages of using the Lazy API over eager execution include

        - automatic query optimization with the query optimizer.
        - ability to process datasets larger than memory using streaming.
        - ability to catch schema errors before data processing.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup

        For this notebook, we are going to work with logs from an Apache/Nginx web server - these logs contain useful information that can be utilized for performance optimization, security monitoring, etc. Such logs comprise of entries that look something like this:

        ```
        10.23.97.15 - - [05/Jul/2024:11:35:05 +0000] "GET /index.html HTTP/1.1" 200 1342 "https://www.example.com" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/528.32 (KHTML, like Gecko) Chrome/19.0.1220.985 Safari/528.32" "-"
        ```

        Different parts of the entry mean different things: 

        - `10.23.97.15` is the client IP address.
        - `- -` represent identity and username of the client, respectively and are typically unused.
        - `05/Jul/2024:11:35:05 +0000` indicates the timestamp for the request.
        - `"GET /index.html HTTP/1.1"` represents the HTTP method, requested resource and the protocol version for HTTP, respectively.
        - `200 1342` mean the response status code and size of the response in bytes, respectively
        - `"https://www.example.com"` is the "referer", or the webpage URL that brought the client to the resource.
        - `"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/528.32 (KHTML, like Gecko) Chrome/19.0.1220.985 Safari/528.32"` is the "User agent" or the details of the client device making the request (including browser version, operating system, etc.)

        Normally, you would get your log files from a server that you have access to. In our case, we will generate fake data to simulate log records. We will simulate 7 days of server activity with 90,000 recorded lines.

        ///Note
        1. If you are interested in the process of generating fake log entries, unhide the code cells immediately below the next one.
        2. You can adjust the size of the dataset by resetting the `num_log_lines` variables to a size of your choice. It may be helpful if the data takes a long time to generate.
        """
    )
    return


@app.cell
def _():
    num_log_lines = 90_000                        # Number of log entries to simulate
    return (num_log_lines,)


@app.cell(hide_code=True)
def _(Faker, datetime, np, random, timedelta, timezone):
    def generate_log_line(*, 
                          faker: Faker, 
                          tz: timezone,
                          sleep: int, 
                          otime: datetime, 
                          rng: np.random.Generator, 
                          resources: list[str], 
                          user_agents: dict[str, float],
                          responses: dict[str, float],
                          verbs: dict[str, float]) -> str:
        """"""
        otime += timedelta(seconds=sleep) if sleep > 0 else timedelta(seconds=random.randint(30, 300))
        dt = otime.strftime('%d/%b/%Y:%H:%M:%S')

        ip, referer = faker.ipv4(), faker.uri()
        vrb = rng.choice(list(verbs.keys()), p=list(verbs.values()))

        uri = rng.choice(resources)
        if uri.find("apps") > 0:
            uri += str(rng.integers(1000, 10000))

        resp = rng.choice(list(responses.keys()), p=list(responses.values()))
        byt = int(rng.normal(5000, 50))
        useragent = rng.choice(list(user_agents.keys()), p=list(user_agents.values()))()
        latency = rng.lognormal(-3, 0.8) + 0.5 * rng.uniform()

        return f'{ip} - - [{dt} {tz}] "{vrb} {uri} HTTP/1.0" {resp} {byt} {latency:.3f} "{referer}" "{useragent}"'
    return (generate_log_line,)


@app.cell(hide_code=True)
def _(Faker, datetime, np, num_log_lines, time):
    tz = datetime.now().strftime('%z')

    faker = Faker()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    otime = datetime.now()

    responses = dict(zip(["200", "404", "500", "301", "403", "502"], 
                         [0.82, 0.04, 0.02, 0.04, 0.05, 0.03]))

    verbs = dict(zip(["GET", "POST", "DELETE", "PUT"], [0.6, 0.1, 0.1, 0.2]))

    resources = ["/list", "/wp-content", "/wp-admin", "/explore", "/search/tag/list", "/app/main/posts",
                 "/posts/posts/explore", "/apps/cart.jsp?appID="]

    user_agents = dict(zip([faker.firefox, faker.chrome, faker.safari, faker.internet_explorer, faker.opera],
                           [0.5, 0.3, 0.1, 0.05, 0.05]))

    sleep = 7 * 24 * 60 * 60 // num_log_lines     # Set interval for 7 days of log data

    rng = np.random.default_rng(seed=int(time.time()))
    return (
        faker,
        otime,
        resources,
        responses,
        rng,
        sleep,
        timestr,
        tz,
        user_agents,
        verbs,
    )


@app.cell(hide_code=True)
def _(
    Generator,
    faker,
    generate_log_line,
    otime,
    re,
    resources,
    responses,
    rng,
    sleep,
    tz,
    user_agents,
    verbs,
):
    pattern = (
        r'^(\S+)'                   # IP address
        r' - [\S+]'                 
        r' \[([^\]]+)\]'            # timestamp
        r' "([A-Z]+)'               # HTTP request code
        r' ([^"]+)'                 # HTTP request resource
        r' HTTP/[^"]+"'          
        r' (\d{3})'                 # Status code (3 digits)
        r' (\S+)'                   # Response size
        r' (\S+)'                   # Latency
        r' "([^"]*)"'               # Referrer
        r' "([^"]*)"'               # User agent
    )

    def generator(log_lines: int) -> \
        Generator[list[str], int, None]:
        for idx in range(log_lines):
            log_line = generate_log_line(tz=tz, sleep=idx*sleep, otime=otime, 
                                  faker=faker, rng=rng, resources=resources, 
                                  user_agents=user_agents, responses=responses, verbs=verbs)
            yield list(re.findall(pattern, log_line)[0])
    return generator, pattern


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since we are generating data using a Python generator, we create a `pl.LazyFrame` directly, but we can start with either a file or an existing `DataFrame`. When using a file, the functions beginning with `pl.scan_` from the Polars API can be used, while in the case of an existing `pl.DataFrame`, we can simply call `.lazy()` to convert it to a `pl.LazyFrame`.

        ///Note
        Depending on your machine, the following cell may take some time to execute.
        """
    )
    return


@app.cell
def _(generator, num_log_lines, pl):
    log_data = pl.LazyFrame(generator(num_log_lines), 
                              schema=["ip", "time", "request_code", 
                                      "request_resource", "status", "size", 
                                      "latency", "referer", "agent"])
    return (log_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Schema

        A schema denotes the names and respective datatypes of columns in a DataFrame or LazyFrame. It can be specified when a DataFrame or LazyFrame is generated (as you may have noticed in the cell creating the LazyFrame above).

        You can see the schema with the .collect_schema method on a DataFrame or LazyFrame.
        """
    )
    return


@app.cell
def _(log_data):
    print(log_data.collect_schema())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Unless specified, Polars defaults to the `pl.String` datatype for all data. This, however, is not the most space or computation efficient form of data storage, so we would like to convert the datatypes of some of the columns in our LazyFrame.

        ///Note
        The data type conversion can also be done by specifying it in the schema when creating the LazyFrame or DataFrame. We are skipping doing this for demonstration.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The Lazy API validates a query pipeline end-to-end for schema consistency and correctness. The checks make sure that if there is a mistake in your query, you can correct it before the data gets processed.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `log_data_erroneous` query below throws an `InvalidOperationError` because Polars finds inconsistencies between the timestamps we parsed from the logs and the timestamp format specified. It turns out that the time stamps in string format still have trailing whitespace which leads to errors during conversion to `datetime[μs]` objects.""")
    return


@app.cell
def _(log_data, pl):
    log_data_erroneous = log_data.with_columns(
        pl.col("time").str.to_datetime("%d/%b/%Y:%H:%M:%S"),
        pl.col("status").cast(pl.Int16),
        pl.col("size").cast(pl.Int32),
        pl.col("latency").cast(pl.Float32)
    )
    return (log_data_erroneous,)


@app.cell
def _(log_data_erroneous):
    log_data_erroneous.collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Polars uses a **query optimizer** to make sure that a query pipeline is executed with the least computational cost (more on this later). In order to be able to do the optimization, the optimizer must know the schema for each step of the pipeline (query plan). For example, if you have a `.pivot` operation somewhere in your pipeline, you are generating new columns based on the data. This is new information unknown to the query optimizer that it cannot work with, and so the lazy API does not support `.pivot` operations. 

        For example, suppose you would like to know how many requests of each kind were received at a given time that were not "POST" requests. For this we would want to create a pivot table as follows, except that it throws an error as the lazy API does not support pivot operations.
        """
    )
    return


@app.cell
def _(log_data, pl):
    (
        log_data.pivot(index="time", on="request_code", 
                   values="status", aggregate_function="len")
                .filter(pl.col("POST").is_null())
                .collect()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As a workaround, we can jump between "lazy mode" and "eager mode" by converting a LazyFrame to a DataFrame just before the unsupported operation (e.g. `.pivot`). We can do this by calling `.collect()` on the LazyFrame. Once done with the "eager mode" operations, we can jump back to "lazy mode" by calling ".lazy()" on the DataFrame!

        As an example, see the fix to the query in the previous cell below:
        """
    )
    return


@app.cell
def _(log_data, pl):
    (
        log_data
            .collect()
            .pivot(index="time", on="request_code", 
                   values="status", aggregate_function="len")
            .lazy()
            .filter(pl.col("POST").is_null())
            .collect()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Query plan""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Polars has a query optimizer that works on a "query plan" to create a computationally efficient query pipeline. It builds the query plan/query graph from the user-specified lazy operations.

        We can understand query graphs with visualization and by printing them as text.

        Say we want to convert the data in our log dataset from `pl.String` more space efficient data types. We also would like to view all "GET" requests that resulted in errors (client side). We build our query first, and then we visualize the query graph using `.show_graph()` and print it using `.request_code()`.
        """
    )
    return


@app.cell
def _(log_data, pl):
    a_query = (
        log_data
            .with_columns(pl.col("status").cast(pl.Int16))
            .with_columns(pl.col("size").cast(pl.Int32))
            .with_columns(pl.col("latency").cast(pl.Float32))
            .with_columns(
                pl.col("time")
                    .str.strip_chars()
                    .str.to_datetime("%d/%b/%Y:%H:%M:%S"),
            )
            .filter((pl.col("status") >= 400) & (pl.col("request_code") == "GET"))
    )
    return (a_query,)


@app.cell
def _(a_query):
    a_query.show_graph()
    return


@app.cell
def _(a_query):
    print(a_query.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Execution""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As mentioned before, Polars builds a query graph by going lazy operation by operation and then optimizes it by running a query optimizer on the graph. This optimized graph is run by default.

        We can execute our query on the full dataset by calling the .collect method on the query. But since this option processes all data in one batch, it is not memory efficient, and can crash if the size of the data exceeds the amount of memory your query can support.

        For fast iterative development running `.collect` on the entire dataset is not a good idea due to slow runtimes. If your dataset is partitioned, you can use a few of them for testing. Another option is to use `.head` to limit the number of records processed, and `.collect` as few times as possible and toward the end of your query, as shown below.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(log_data, pl):
    (
        log_data
            .head(n=100)
            .with_columns(pl.col("status").cast(pl.Int16))
            .with_columns(pl.col("size").cast(pl.Int32))
            .with_columns(pl.col("latency").cast(pl.Float32))
            .with_columns(
                pl.col("time")
                    .str.strip_chars()
                    .str.to_datetime("%d/%b/%Y:%H:%M:%S"),
            )
            .filter((pl.col("status") >= 400) & (pl.col("request_code") == "GET"))
            .collect()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For large datasets Polars supports streaming mode by collecting data in batches. Streaming mode can be used by passing the keyword `engine="streaming"` into the `collect` method.""")
    return


@app.cell
def _(a_query):
    a_query.collect(engine="streaming")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Optimizations

        The lazy API runs a query optimizer on every Polars query. To do this, first it builds a non-optimized plan with the set of steps in the order they were specified by the user. Then it checks for optimization opportunities within the plan and reorders operations following specific rules to create an optimized query plan. Some of them are executed up front, others are determined just in time as the materialized data comes in. For the query that we built before and saw the query graph, we can view the unoptimized and optimized versions below.
        """
    )
    return


@app.cell
def _(a_query):
    a_query.show_graph(optimized=False)
    return


@app.cell
def _(a_query):
    a_query.show_graph()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""One difference between the optimized and the unoptimized versions above is that all of the datatype cast operations except for the conversion of the `"status"` column to `pl.Int16` are performed at the end together. Also, the `filter()` operation is "pushed down" the graph, but after the datatype cast operation for `"status"`. This is called **predicate pushdown**, and the lazy API optimizes the query graph for filters to be performed as early as possible. Since the datatype coercion makes the filter operation more efficient, the graph preserves its order to be before the filter.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Sources and Sinks""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For data sources like Parquets, CSVs, etc, the lazy API provides `scan_*` (`scan_parquet`, `scan_csv`, etc.) to lazily read in the data into LazyFrames. If queries are chained to the `scan_*` method, Polars will run the usual query optimizations and delay execution until the query is collected. An added benefit of chaining queries to `scan_*` operations is that the "scanners" can skip reading columns and rows that aren't required. This is helpful when streaming large datasets as well, as rows are processed in batches before the entire file is read.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The results of a query from a lazyframe can be saved in streaming mode using `sink_*` (e.g. `sink_parquet`) functions. Sinks support saving data to disk or cloud, and are especially helpful with large datasets. The data being sunk can also be partitioned into multiple files if needed, after specifying a suitable partitioning strategy, as shown below.""")
    return


@app.cell
def _(a_query, pl):
    (
        a_query
            .sink_parquet(
                pl.PartitionMaxSize(
                    "log_data_filtered_{part}.parquet",
                    max_size=1_000
                )
            )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also write to multiple sinks at the same time. We just need to specify two separate lazy sinks and combine them by calling `pl.collect_all` and mentioning both sinks.""")
    return


@app.cell
def _(a_query, pl):
    _q1 = a_query.sink_parquet("log_data_filtered.parquet", lazy=True)
    _q2 = a_query.sink_ipc("log_data_filtered.ipc", lazy=True)
    pl.collect_all([_q1, _q2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        1. Polars [documentation](https://docs.pola.rs/user-guide/lazy/)
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "beautifulsoup4==4.13.3",
#     "httpx==0.28.1",
#     "marimo",
#     "nest-asyncio==1.6.0",
#     "numba==0.61.0",
#     "numpy==2.1.3",
#     "polars==1.24.0",
# ]
# ///

import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # User-Defined Functions

        _By [Péter Ferenc Gyarmati](http://github.com/peter-gy)_.

        Throughout the previous chapters, you've seen how Polars provides a comprehensive set of built-in expressions for flexible data transformation.  But what happens when you need something *more*? Perhaps your project has unique requirements, or you need to integrate functionality from an external Python library. This is where User-Defined Functions (UDFs) come into play, allowing you to extend Polars with your own custom logic.

        In this chapter, we'll weigh the performance trade-offs of UDFs, pinpoint situations where they're truly beneficial, and explore different ways to effectively incorporate them into your Polars workflows. We'll walk through a complete, practical example.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## ⚖️ The Cost of UDFs

        > Performance vs. Flexibility

        Polars' built-in expressions are highly optimized for speed and parallel processing. User-defined functions (UDFs), however, introduce a significant performance overhead because they rely on standard Python code, which often runs in a single thread and bypasses Polars' logical optimizations. Therefore, always prioritize native Polars operations *whenever possible*.

        However, UDFs become inevitable when you need to:

        -  **Integrate external libraries:**  Use functionality not directly available in Polars.
        -  **Implement custom logic:** Handle complex transformations that can't be easily expressed with Polars' built-in functions.

        Let's dive into a real-world project where UDFs were the only way to get the job done, demonstrating a scenario where native Polars expressions simply weren't sufficient.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 📊 Project Overview

        > Scraping and Analyzing Observable Notebook Statistics

        If you're into data visualization, you've probably seen [D3.js](https://d3js.org/) and [Observable Plot](https://observablehq.com/plot/). Both have extensive galleries showcasing amazing visualizations. Each gallery item is a standalone [Observable notebook](https://observablehq.com/documentation/notebooks/), with metrics like stars, comments, and forks – indicators of popularity. But getting and analyzing these statistics directly isn't straightforward. We'll need to scrape the web.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack(
        [
            mo.image(
                "https://minio.peter.gy/static/assets/marimo/learn/polars/14_d3-gallery.png?0",
                width=600,
                caption="Screenshot of https://observablehq.com/@d3/gallery",
            ),
            mo.image(
                "https://minio.peter.gy/static/assets/marimo/learn/polars/14_plot-gallery.png?0",
                width=600,
                caption="Screenshot of https://observablehq.com/@observablehq/plot-gallery",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our goal is to use Polars UDFs to fetch the HTML content of these gallery pages. Then, we'll use the `BeautifulSoup` Python library to parse the HTML and extract the relevant metadata.  After some data wrangling with native Polars expressions, we'll have a DataFrame listing each visualization notebook. Then, we'll use another UDF to retrieve the number of likes, forks, and comments for each notebook. Finally, we will create our own high-performance UDF to implement a custom notebook ranking scheme. This will involve multiple steps, showcasing different UDF approaches.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid('''
    graph LR;
        url_df --> |"UDF: Fetch HTML"| html_df
        html_df --> |"UDF: Parse with BeautifulSoup"| parsed_html_df
        parsed_html_df --> |"Native Polars: Extract Data"| notebooks_df
        notebooks_df --> |"UDF: Get Notebook Stats"| notebook_stats_df
        notebook_stats_df --> |"Numba UDF: Compute Popularity"| notebook_popularity_df
    ''')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our starting point, `url_df`, is a simple DataFrame with a single `url` column containing the URLs of the D3 and Observable Plot gallery notebooks.""")
    return


@app.cell(hide_code=True)
def _(pl):
    url_df = pl.from_dict(
        {
            "url": [
                "https://observablehq.com/@d3/gallery",
                "https://observablehq.com/@observablehq/plot-gallery",
            ]
        }
    )
    url_df
    return (url_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🔂 Element-Wise UDFs

        > Processing Value by Value

        The most common way to use UDFs is to apply them element-wise.  This means our custom function will execute for *each individual row* in a specified column.  Our first task is to fetch the HTML content for each URL in `url_df`.

        We'll define a Python function that takes a `url` (a string) as input, uses the `httpx` library (an HTTP client) to fetch the content, and returns the HTML as a string. We then integrate this function into Polars using the [`map_elements`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.map_elements.html) expression.

        You'll notice we have to explicitly specify the `return_dtype`.  This is *crucial*.  Polars doesn't automatically know what our custom function will return.  We're responsible for defining the function's logic and, therefore, its output type. By providing the `return_dtype`, we help Polars maintain its internal representation of the DataFrame's schema, enabling query optimization. Think of it as giving Polars a "heads-up" about the data type it should expect.
        """
    )
    return


@app.cell(hide_code=True)
def _(httpx, pl, url_df):
    html_df = url_df.with_columns(
        html=pl.col("url").map_elements(
            lambda url: httpx.get(url).text,
            return_dtype=pl.String,
        )
    )
    html_df
    return (html_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, `html_df` holds the HTML for each URL.  We need to parse it. Again, a UDF is the way to go. Parsing HTML with native Polars expressions would be a nightmare! Instead, we'll use the [`beautifulsoup4`](https://pypi.org/project/beautifulsoup4/) library, a standard tool for this.

        These Observable pages are built with [Next.js](https://nextjs.org/), which helpfully serializes page properties as JSON within the HTML. This simplifies our UDF: we'll extract the raw JSON from the `<script id="__NEXT_DATA__" type="application/json">` tag. We'll use [`map_elements`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.map_elements.html) again.  For clarity, we'll define this UDF as a named function, `extract_nextjs_data`, since it's a bit more complex than a simple HTTP request.
        """
    )
    return


@app.cell(hide_code=True)
def _(BeautifulSoup):
    def extract_nextjs_data(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        script_tag = soup.find("script", id="__NEXT_DATA__")
        return script_tag.text
    return (extract_nextjs_data,)


@app.cell(hide_code=True)
def _(extract_nextjs_data, html_df, pl):
    parsed_html_df = html_df.select(
        "url",
        next_data=pl.col("html").map_elements(
            extract_nextjs_data,
            return_dtype=pl.String,
        ),
    )
    parsed_html_df
    return (parsed_html_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""With some data wrangling of the raw JSON (using *native* Polars expressions!), we get `notebooks_df`, containing the metadata for each notebook.""")
    return


@app.cell(hide_code=True)
def _(parsed_html_df, pl):
    notebooks_df = (
        parsed_html_df.select(
            "url",
            # We extract the content of every cell present in the gallery notebooks
            cell=pl.col("next_data")
            .str.json_path_match("$.props.pageProps.initialNotebook.nodes")
            .str.json_decode()
            .list.eval(pl.element().struct.field("value")),
        )
        # We want one row per cell
        .explode("cell")
        # Only keep categorized notebook listing cells starting with H3
        .filter(pl.col("cell").str.starts_with("### "))
        # Split up the cells into [heading, description, config] sections
        .with_columns(pl.col("cell").str.split("\n\n"))
        .select(
            gallery_url="url",
            # Text after the '### ' heading, ignore '<!--' comments'
            category=pl.col("cell").list.get(0).str.extract(r"###\s+(.*?)(?:\s+<!--.*?-->|$)"),
            # Paragraph after heading
            description=pl.col("cell")
            .list.get(1)
            .str.strip_chars(" ")
            .str.replace_all("](/", "](https://observablehq.com/", literal=True),
            # Parsed notebook config from ${preview([{...}])}
            notebooks=pl.col("cell")
            .list.get(2)
            .str.strip_prefix("${previews([")
            .str.strip_suffix("]})}")
            .str.strip_chars(" \n")
            .str.split("},")
            # Simple regex-based attribute extraction from JS/JSON objects like
            # ```js
            # {
            #   path: "@d3/spilhaus-shoreline-map",
            #   "thumbnail": "66a87355e205d820...",
            #   title: "Spilhaus shoreline map",
            #   "author": "D3"
            # }
            # ```
            .list.eval(
                pl.struct(
                    *(
                        pl.element()
                        .str.extract(f'(?:"{key}"|{key})\s*:\s*"([^"]*)"')
                        .alias(key)
                        for key in ["path", "thumbnail", "title"]
                    )
                )
            ),
        )
        .explode("notebooks")
        .unnest("notebooks")
        .filter(pl.col("path").is_not_null())
        # Final projection to end up with directly usable values
        .select(
            pl.concat_str(
                [
                    pl.lit("https://static.observableusercontent.com/thumbnail/"),
                    "thumbnail",
                    pl.lit(".jpg"),
                ],
            ).alias("notebook_thumbnail_src"),
            "category",
            "title",
            "description",
            pl.concat_str(
                [pl.lit("https://observablehq.com"), "path"], separator="/"
            ).alias("notebook_url"),
        )
    )
    notebooks_df
    return (notebooks_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 📦 Batch-Wise UDFs

        > Processing Entire Series

        `map_elements` calls the UDF for *each row*. Fine for our tiny, two-rows-tall `url_df`. But `notebooks_df` has almost 400 rows! Individual HTTP requests for each would be painfully slow.

        We want stats for each notebook in `notebooks_df`. To avoid sequential requests, we'll use Polars' [`map_batches`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.map_batches.html). This lets us process an *entire Series* (a column) at once.

        Our UDF, `fetch_html_batch`, will take a *Series* of URLs and use `asyncio` to make concurrent requests – a huge performance boost.
        """
    )
    return


@app.cell(hide_code=True)
def _(Iterable, asyncio, httpx, mo):
    async def _fetch_html_batch(urls: Iterable[str]) -> tuple[str, ...]:
        async with httpx.AsyncClient(timeout=15) as client:
            res = await asyncio.gather(*(client.get(url) for url in urls))
            return tuple((r.text for r in res))


    @mo.cache
    def fetch_html_batch(urls: Iterable[str]) -> tuple[str, ...]:
        return asyncio.run(_fetch_html_batch(urls))
    return (fetch_html_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""
    Since `fetch_html_batch` is a pure Python function and performs multiple network requests, it's a good candidate for caching. We use [`mo.cache`](https://docs.marimo.io/api/caching/#marimo.cache) to avoid redundant requests to the same URL. This is a simple way to improve performance without modifying the core logic.
    """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo, notebooks_df):
    category = mo.ui.dropdown(
        notebooks_df.sort("category").get_column("category"),
        value="Maps",
    )
    return (category,)


@app.cell(hide_code=True)
def _(category, extract_nextjs_data, fetch_html_batch, notebooks_df, pl):
    notebook_stats_df = (
        # Setting filter upstream to limit number of concurrent HTTP requests
        notebooks_df.filter(category=category.value)
        .with_columns(
            notebook_html=pl.col("notebook_url")
            .map_batches(fetch_html_batch, return_dtype=pl.List(pl.String))
            .explode()
        )
        .with_columns(
            notebook_data=pl.col("notebook_html")
            .map_elements(
                extract_nextjs_data,
                return_dtype=pl.String,
            )
            .str.json_path_match("$.props.pageProps.initialNotebook")
            .str.json_decode()
        )
        .drop("notebook_html")
        .with_columns(
            *[
                pl.col("notebook_data").struct.field(key).alias(key)
                for key in ["likes", "forks", "comments", "license"]
            ]
        )
        .drop("notebook_data")
        .with_columns(pl.col("comments").list.len())
        .select(
            pl.exclude("description", "notebook_url"),
            "description",
            "notebook_url",
        )
        .sort("likes", descending=True)
    )
    return (notebook_stats_df,)


@app.cell(hide_code=True)
def _(mo, notebook_stats_df):
    notebooks = mo.ui.table(notebook_stats_df, selection='single', initial_selection=[2], page_size=5)
    notebook_height = mo.ui.slider(start=400, stop=2000, value=825, step=25, show_value=True, label='Notebook Height')
    return notebook_height, notebooks


@app.cell(hide_code=True)
def _():
    def nb_iframe(notebook_url: str, height=825) -> str:
        embed_url = notebook_url.replace(
            "https://observablehq.com", "https://observablehq.com/embed"
        )
        return f'<iframe width="100%" height="{height}" frameborder="0" src="{embed_url}?cell=*"></iframe>'
    return (nb_iframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now that we have access to notebook-level statistics, we can rank the visualizations by the number of likes they received & display them interactively.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout("💡 Explore the visualizations by paging through the table below and selecting any of its rows.")
    return


@app.cell(hide_code=True)
def _(category, mo, nb_iframe, notebook_height, notebooks):
    notebook = notebooks.value.to_dicts()[0]
    mo.vstack(
        [
            mo.hstack([category, notebook_height]),
            notebooks,
            mo.md(f"{notebook['description']}"),
            mo.md('---'),
            mo.md(nb_iframe(notebook["notebook_url"], notebook_height.value)),
        ]
    )
    return (notebook,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## ⚙️ Row-Wise UDFs

        > Accessing All Columns at Once

        Sometimes, you need to work with *all* columns of a row at once.  This is where [`map_rows`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.map_rows.html) comes in. It operates directly on the DataFrame, passing each row to your UDF *as a tuple*.

        Below, `create_notebook_summary` takes a row from `notebook_stats_df` (as a tuple) and returns a formatted Markdown string summarizing the notebook's key stats.  We're essentially reducing the DataFrame to a single column.  While this *could* be done with native Polars expressions, it would be much more cumbersome. This example demonstrates a case where a row-wise UDF simplifies the code, even if the underlying operation isn't inherently complex.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    def create_notebook_summary(row: tuple) -> str:
        (
            thumbnail_src,
            category,
            title,
            likes,
            forks,
            comments,
            license,
            description,
            notebook_url,
        ) = row
        return (
            f"""
    ### [{title}]({notebook_url})

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0;">
        <div>⭐ <strong>Likes:</strong> {likes}</div>
        <div>↗️ <strong>Forks:</strong> {forks}</div>
        <div>💬 <strong>Comments:</strong> {comments}</div>
        <div>⚖️ <strong>License:</strong> {license}</div>
    </div>

    <a href="{notebook_url}" target="_blank">
        <img src="{thumbnail_src}" style="height: 300px;" />
    <a/>
    """.strip('\n')
        )
    return (create_notebook_summary,)


@app.cell(hide_code=True)
def _(create_notebook_summary, notebook_stats_df, pl):
    notebook_summary_df = notebook_stats_df.map_rows(
        create_notebook_summary,
        return_dtype=pl.String,
    ).rename({"map": "summary"})
    notebook_summary_df.head(1)
    return (notebook_summary_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout("💡 You can explore individual notebook statistics through the carousel. Discover the visualization's source code by clicking the notebook title or the thumbnail.")
    return


@app.cell(hide_code=True)
def _(mo, notebook_summary_df):
    mo.carousel(
        [
            mo.lazy(mo.md(summary))
            for summary in notebook_summary_df.get_column("summary")
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🚀 Higher-performance UDFs

        > Leveraging Numba to Make Python Fast

        Python code doesn't *always* mean slow code. While UDFs *often* introduce performance overhead, there are exceptions. NumPy's universal functions ([`ufuncs`](https://numpy.org/doc/stable/reference/ufuncs.html)) and generalized universal functions ([`gufuncs`](https://numpy.org/neps/nep-0005-generalized-ufuncs.html)) provide high-performance operations on NumPy arrays, thanks to low-level implementations.

        But NumPy's built-in functions are predefined. We can't easily use them for *custom* logic. Enter [`numba`](https://numba.pydata.org/).  Numba is a just-in-time (JIT) compiler that translates Python functions into optimized machine code *at runtime*. It provides decorators like [`numba.guvectorize`](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-guvectorize-decorator) that let us create our *own* high-performance `gufuncs` – *without* writing low-level code!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's create a custom popularity metric to rank notebooks, considering likes, forks, *and* comments (not just likes).  We'll define `weighted_popularity_numba`, decorated with `@numba.guvectorize`.  The decorator arguments specify that we're taking three integer vectors of length `n` and returning a float vector of length `n`.

        The weighted popularity score for each notebook is calculated using the following formula:

        $$
        \begin{equation}
        \text{score}_i = w_l \cdot l_i^{f} + w_f \cdot f_i^{f} + w_c \cdot c_i^{f}
        \end{equation}
        $$

        with:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, non_linear_factor, weight_comments, weight_forks, weight_likes):
    mo.md(rf"""
    | Symbol | Description |
    |--------|-------------|
    | $\text{{score}}_i$ | Popularity score for the *i*-th notebook |
    | $w_l = {weight_likes.value}$ | Weight for likes |
    | $l_i$ | Number of likes for the *i*-th notebook |
    | $w_f = {weight_forks.value}$ | Weight for forks |
    | $f_i$ | Number of forks for the *i*-th notebook |
    | $w_c = {weight_comments.value}$ | Weight for comments |
    | $c_i$ | Number of comments for the *i*-th notebook |
    | $f = {non_linear_factor.value}$ | Non-linear factor (exponent) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    weight_likes = mo.ui.slider(
        start=0.1,
        stop=1,
        value=0.5,
        step=0.1,
        show_value=True,
        label="⭐ Weight for Likes",
    )
    weight_forks = mo.ui.slider(
        start=0.1,
        stop=1,
        value=0.3,
        step=0.1,
        show_value=True,
        label="↗️ Weight for Forks",
    )
    weight_comments = mo.ui.slider(
        start=0.1,
        stop=1,
        value=0.5,
        step=0.1,
        show_value=True,
        label="💬 Weight for Comments",
    )
    non_linear_factor = mo.ui.slider(
        start=1,
        stop=2,
        value=1.2,
        step=0.1,
        show_value=True,
        label="🎢 Non-Linear Factor",
    )
    return non_linear_factor, weight_comments, weight_forks, weight_likes


@app.cell(hide_code=True)
def _(
    non_linear_factor,
    np,
    numba,
    weight_comments,
    weight_forks,
    weight_likes,
):
    w_l = weight_likes.value
    w_f = weight_forks.value
    w_c = weight_comments.value
    nlf = non_linear_factor.value


    @numba.guvectorize(
        [(numba.int64[:], numba.int64[:], numba.int64[:], numba.float64[:])],
        "(n), (n), (n) -> (n)",
    )
    def weighted_popularity_numba(
        likes: np.ndarray,
        forks: np.ndarray,
        comments: np.ndarray,
        out: np.ndarray,
    ):
        for i in range(likes.shape[0]):
            out[i] = (
                w_l * (likes[i] ** nlf)
                + w_f * (forks[i] ** nlf)
                + w_c * (comments[i] ** nlf)
            )
    return nlf, w_c, w_f, w_l, weighted_popularity_numba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We apply our JIT-compiled UDF using `map_batches`, as before.  The key is that we're passing entire columns directly to `weighted_popularity_numba`. Polars and Numba handle the conversion to NumPy arrays behind the scenes. This direct integration is a major benefit of using `guvectorize`.""")
    return


@app.cell(hide_code=True)
def _(notebook_stats_df, pl, weighted_popularity_numba):
    notebook_popularity_df = (
        notebook_stats_df.select(
            pl.col("notebook_thumbnail_src").alias("thumbnail"),
            "title",
            "likes",
            "forks",
            "comments",
            popularity=pl.struct(["likes", "forks", "comments"]).map_batches(
                lambda obj: weighted_popularity_numba(
                    obj.struct.field("likes"),
                    obj.struct.field("forks"),
                    obj.struct.field("comments"),
                ),
                return_dtype=pl.Float64,
            ),
            url="notebook_url",
        )
    )
    return (notebook_popularity_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout("💡 Adjust the hyperparameters of the popularity ranking UDF. How do the weights and non-linear factor affect the notebook rankings?")
    return


@app.cell(hide_code=True)
def _(
    mo,
    non_linear_factor,
    notebook_popularity_df,
    weight_comments,
    weight_forks,
    weight_likes,
):
    mo.vstack(
        [
            mo.hstack([weight_likes, weight_forks]),
            mo.hstack([weight_comments, non_linear_factor]),
            notebook_popularity_df,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As the slope chart below demonstrates, this new ranking strategy significantly changes the notebook order, as it considers forks and comments, not just likes.""")
    return


@app.cell(hide_code=True)
def _(alt, notebook_popularity_df, pl):
    notebook_ranks_df = (
        notebook_popularity_df.sort("likes", descending=True)
        .with_row_index("rank_by_likes")
        .with_columns(pl.col("rank_by_likes") + 1)
        .sort("popularity", descending=True)
        .with_row_index("rank_by_popularity")
        .with_columns(pl.col("rank_by_popularity") + 1)
        .select("thumbnail", "title", "rank_by_popularity", "rank_by_likes")
        .unpivot(
            ["rank_by_popularity", "rank_by_likes"],
            index="title",
            variable_name="strategy",
            value_name="rank",
        )
    )

    # Slope chart to visualize rank differences by strategy
    lines = notebook_ranks_df.plot.line(
        x="strategy:O",
        y="rank:Q",
        color="title:N",
    )
    points = notebook_ranks_df.plot.point(
        x="strategy:O",
        y="rank:Q",
        color=alt.Color("title:N", legend=None),
        fill="title:N",
    )
    (points + lines).properties(width=400)
    return lines, notebook_ranks_df, points


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## ⏱️ Quantifying the Overhead

        > UDF Performance Comparison

        To truly understand the performance implications of using UDFs, let's conduct a benchmark.  We'll create a DataFrame with random numbers and perform the same numerical operation using four different methods:

        1. **Native Polars:** Using Polars' built-in expressions.
        2. **`map_elements`:**  Applying a Python function element-wise.
        3. **`map_batches`:** **Applying** a Python function to the entire Series.
        4. **`map_batches` with Numba:** Applying a JIT-compiled function to batches, similar to a generalized universal function.

        We'll use a simple, but non-trivial, calculation:  `result = (x * 2.5 + 5) / (x + 1)`. This involves multiplication, addition, and division, giving us a realistic representation of a common numerical operation. We'll use the `timeit` module, to accurately measure execution times over multiple trials.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout("💡 Tweak the benchmark parameters to explore how execution times change with different sample sizes and trial counts. Do you notice anything surprising as you decrease the number of samples?")
    return


@app.cell(hide_code=True)
def _(benchmark_plot, mo, num_samples, num_trials):
    mo.vstack(
        [
            mo.hstack([num_samples, num_trials]),
            mo.md(
                f"""---
    Performance comparison over **{num_trials.value:,} trials** with **{num_samples.value:,} samples**.

    > Lower execution times are better.
    """
            ),
            benchmark_plot,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As anticipated, the `Batch-Wise UDF (Python)` and `Element-Wise UDF` exhibit significantly worse performance, essentially acting as pure-Python for-each loops.  

        However, when Python serves as an interface to lower-level, high-performance libraries, we observe substantial improvements. The `Batch-Wise UDF (NumPy)` lags behind both `Batch-Wise UDF (Numba)` and `Native Polars`, but it still represents a considerable improvement over pure-Python UDFs due to its vectorized computations. 

        Numba's Just-In-Time (JIT) compilation delivers a dramatic performance boost, achieving speeds comparable to native Polars expressions. This demonstrates that UDFs, particularly when combined with tools like Numba, don't inevitably lead to bottlenecks in numerical computations.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    num_samples = mo.ui.slider(
        start=1_000,
        stop=1_000_000,
        value=250_000,
        step=1000,
        show_value=True,
        debounce=True,
        label="Number of Samples",
    )
    num_trials = mo.ui.slider(
        start=50,
        stop=1_000,
        value=100,
        step=50,
        show_value=True,
        debounce=True,
        label="Number of Trials",
    )
    return num_samples, num_trials


@app.cell(hide_code=True)
def _(np, num_samples, pl):
    rng = np.random.default_rng(42)
    sample_df = pl.from_dict({"x": rng.random(num_samples.value)})
    return rng, sample_df


@app.cell(hide_code=True)
def _(np, num_trials, numba, pl, sample_df, timeit):
    def run_native():
        sample_df.with_columns(
            result_native=(pl.col("x") * 2.5 + 5) / (pl.col("x") + 1)
        )


    def _calculate_elementwise(x: float) -> float:
        return (x * 2.5 + 5) / (x + 1)


    def run_map_elements():
        sample_df.with_columns(
            result_map_elements=pl.col("x").map_elements(
                _calculate_elementwise,
                return_dtype=pl.Float64,
            )
        )


    def _calculate_batchwise_numpy(x_series: pl.Series) -> pl.Series:
        x_array = x_series.to_numpy()
        result_array = (x_array * 2.5 + 5) / (x_array + 1)
        return pl.Series(result_array)


    def run_map_batches_numpy():
        sample_df.with_columns(
            result_map_batches_numpy=pl.col("x").map_batches(
                _calculate_batchwise_numpy,
                return_dtype=pl.Float64,
            )
        )


    def _calculate_batchwise_python(x_series: pl.Series) -> pl.Series:
        x_array = x_series.to_list()
        result_array = [_calculate_elementwise(x) for x in x_array]
        return pl.Series(result_array)


    def run_map_batches_python():
        sample_df.with_columns(
            result_map_batches_python=pl.col("x").map_batches(
                _calculate_batchwise_python,
                return_dtype=pl.Float64,
            )
        )


    @numba.guvectorize([(numba.float64[:], numba.float64[:])], "(n) -> (n)")
    def _calculate_batchwise_numba(x: np.ndarray, out: np.ndarray):
        for i in range(x.shape[0]):
            out[i] = (x[i] * 2.5 + 5) / (x[i] + 1)


    def run_map_batches_numba():
        sample_df.with_columns(
            result_map_batches_numba=pl.col("x").map_batches(
                _calculate_batchwise_numba,
                return_dtype=pl.Float64,
            )
        )


    def time_method(callable_name: str, number=num_trials.value) -> float:
        fn = globals()[callable_name]
        return timeit.timeit(fn, number=number)
    return (
        run_map_batches_numba,
        run_map_batches_numpy,
        run_map_batches_python,
        run_map_elements,
        run_native,
        time_method,
    )


@app.cell(hide_code=True)
def _(alt, pl, time_method):
    benchmark_df = pl.from_dicts(
        [
            {
                "title": "Native Polars",
                "callable_name": "run_native",
            },
            {
                "title": "Element-Wise UDF",
                "callable_name": "run_map_elements",
            },
            {
                "title": "Batch-Wise UDF (NumPy)",
                "callable_name": "run_map_batches_numpy",
            },
            {
                "title": "Batch-Wise UDF (Python)",
                "callable_name": "run_map_batches_python",
            },
            {
                "title": "Batch-Wise UDF (Numba)",
                "callable_name": "run_map_batches_numba",
            },
        ]
    ).with_columns(
        time=pl.col("callable_name").map_elements(
            time_method, return_dtype=pl.Float64
        )
    )

    benchmark_plot = benchmark_df.plot.bar(
        x=alt.X("title:N", title="Method", sort="-y"),
        y=alt.Y("time:Q", title="Execution Time (s)", axis=alt.Axis(format=".3f")),
    ).properties(width=400)
    return benchmark_df, benchmark_plot


@app.cell(hide_code=True)
def _():
    import asyncio
    import timeit
    from typing import Iterable

    import altair as alt
    import httpx
    import marimo as mo
    import nest_asyncio
    import numba
    import numpy as np
    from bs4 import BeautifulSoup

    import polars as pl

    # Fixes RuntimeError: asyncio.run() cannot be called from a running event loop
    nest_asyncio.apply()
    return (
        BeautifulSoup,
        Iterable,
        alt,
        asyncio,
        httpx,
        mo,
        nest_asyncio,
        np,
        numba,
        pl,
        timeit,
    )


if __name__ == "__main__":
    app.run()

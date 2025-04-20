# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy==2.2.3",
#     "plotly[express]==6.0.0",
#     "polars==1.27.1",
#     "statsmodels==0.14.4",
# ]
# ///

import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Reactive Plots

        _By [etrotta](https://github.com/etrotta)._

        This tutorial covers Data Visualisation basics using marimo, [polars](https://docs.pola.rs/) and [plotly](https://plotly.com/python/plotly-express/).
        It shows how to load data, explore and visualise it, then use User Interface elements (including the plots themselves) to filter and select data for more refined analysis.

        We will be using a [Spotify Tracks dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset). Before you write any code yourself, I recommend taking some time to understand the data you're working with, from which columns are available to what are their possible values, as well as more abstract details such as the scope, coverage and intended uses of the dataset.

        Note that this dataset does not contains data about ***all***  tracks, you can try using a larger dataset such as [bigdata-pw/Spotify](https://huggingface.co/datasets/bigdata-pw/Spotify), but I'm sticking with the smaller one to keep the notebook size manageable for most users.
        """
    )
    return


@app.cell
def _(pl):
    # You can read directly from the Hugging Face dataset, in which case polars will only read the necessary data:
    repo_id, branch, file_path = (
        "maharshipandya/spotify-tracks-dataset",
        "~parquet",
        "default/train/0000.parquet",
    )
    URL = f"hf://datasets/{repo_id}@{branch}/{file_path}"
    lz = pl.scan_parquet(URL)
    # Or save to a local file first if you want to avoid downloading it each time you run:
    # file_path = "spotify-tracks.parquet"
    # lz = pl.scan_parquet(file_path)
    return URL, branch, file_path, lz, repo_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        You should always take a look at the data you are working on before actually doing any operations on it - for data coming from sources such as HuggingFace or Kaggle you can preview it via their websites, and optionally filter or do some transformations before downloading.

        The [Polars Lazy API](https://docs.pola.rs/user-guide/lazy/) allows for you define operations before loading the data, and polars will optimize the plan in order to avoid doing unnecessary operations or loading data we do not care about.

        Let's say that looking at the dataset's preview in the Data Viewer, we decided we do not want the Unnamed column (which appears to be the row index), nor do we care about the original ID, and we only want non-explicit tracks.
        """
    )
    return


@app.cell
def _(lz, pl):
    df = (
        lz
        # Filter data we consider relevant (somewhat arbitrary in this example)
        .filter(pl.col("explicit") == False)
        .drop("Unnamed: 0", "track_id", "explicit")
        .with_columns(
            # Perform whichever transformations you want  (again somewhat arbitrary in this example)
            # Convert the duration from milliseconds to seconds (int)
            pl.col("duration_ms").floordiv(1_000).alias("duration_seconds"),
            # Convert the popularity from an integer 0 ~ 100 to a percentage 0 ~ 1.0
            pl.col("popularity").truediv(100),
        )
        # lastly, download (if needed) and collect into memory
        .collect()
    )
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When you start exploring a dataset, some of the first things to do may include:

        - investigating any values that seem weird
        - verifying if there could be issues in the data
        - checking for potential bugs in our pipelines
        - ensuring you understand the data correctly, including its relationships and edge cases

        For example, the "min" value for the duration column is zero, and the max is over an hour. Why is that?
        """
    )
    return


@app.cell(disabled=True)
def _(df, pl):
    # We *could* just filter some of the rows and look at them as a table, for example...
    pl.concat([df.sort("duration_ms").head(5), df.sort("duration_ms", descending=True).head(5)])
    # But creating a visualisation for this helps paint the full picture of how the data is distributed, rather than focusing *only* on some outiers
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For this Notebook we will be using [plotly](https://plotly.com/python), but Marimo also [supports other plotting libraries](https://docs.marimo.io/guides/working_with_data/plotting/).

        Let's visualize it using a [bar chart](https://plotly.com/python/bar-charts/) and get a feel for which region makes sense to focus on for our analysis
        """
    )
    return


@app.cell
def _(df, mo, px):
    duration_counts = df.group_by("duration_seconds").len("count")
    fig = px.bar(duration_counts, x="duration_seconds", y="count")
    fig.update_layout(selectdirection="h")
    plot = mo.ui.plotly(fig)
    plot
    return duration_counts, fig, plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Note how there are a few outliers with extremely little duration (less than 2 minutes) and a few with extremely long duration (more than 6 minutes)

        You can select a region in the graph by clicking and dragging, which can later be used to filter or transform data. In this Notebook we set a default if there is no selection, but you should try selecting a region yourself.

        We will focus on those within that middle ground from around 120 seconds to 360 seconds, but you can play around with it a bit and see how the results change if you move the Selection region. Perhaps you can even find some Classical songs?
        """
    )
    return


@app.cell
def _(pl, plot):
    # The format of plot.value may vary depending on which kind of plot you are working with, let's see what we have for this case:
    pl.DataFrame(plot.value)
    return


@app.cell
def _(df, get_extremes, pl, plot):
    # Now, we want to filter to only include tracks whose duration falls inside of our selection - we will need to first identify the extremes, then filter based on them
    min_dur, max_dur = get_extremes(
        plot.value, col="duration_seconds", defaults_if_missing=(120, 360)
    )  # Utility function defined in the bottom of the Notebook
    # Calculate how many we are keeping vs throwing away with the filter
    duration_in_range = pl.col("duration_seconds").is_between(min_dur, max_dur)
    print(
        f"Filtering to keep rows between {min_dur}s and {max_dur}s duration - Throwing away {df.select(1 - duration_in_range.mean()).item():.2%} of the rows"
    )

    # Actually apply the filter
    filtered_duration = df.filter(duration_in_range)
    filtered_duration
    return duration_in_range, filtered_duration, max_dur, min_dur


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now that our data is 'clean', let's start coming up with and answering some questions about it. Some examples:

        - Which tracks or artists are the most popular? (Both globally as well as for each genre)
        - Which genres are the most popular? The loudest?
        - What are some common combinations of different artists? 
        - What can we infer anything based on the track's title or artist name?
        - How popular is some specific song you like?
        - How much does the mode and key affect other attributes?
        - Can you classify a song's genre based on its attributes?

        For brevity, we will not explore all of them - feel free to try some of the others yourself, or go more in deep in the explored ones.
        Make sure to come up with some questions of your own and explore them as well!
        """
    )
    return


@app.cell(hide_code=True)
def _(filter_genre, filtered_duration, mo, pl):
    most_popular_artists = (
        filtered_duration.lazy()
        # If you saw the Dataset description or looked closely at the Artists column you may notice there are some rows with multiple artists separated by ;;. We will have to separate each of these.
        .with_columns(pl.col("artists").str.split(";"))
        # Similarly to the utility function you saw before, filter_genre is also defined in a later cell.
        # While developing, you can add things out of order then go back to old cells and edit them
        # it's up to you whenever to put them in whichever order makes the most sense to you.
        .filter(True if filter_genre.value is None else pl.col("track_genre").eq(filter_genre.value))
        .explode("artists")
        .group_by("artists")
        .agg(
            # How to aggregate it is also a question,
            # Do we take the sum of each of their songs popularity?
            # Do we just take their most popular song?
            # Do we take an average of their songs popularity?
            # We'll proceed with the average of their top 10 most popular songs for now,
            # but that is something you may want to modify and experiment with, or ask for input from stakeholders in real problems.
            pl.col("popularity").top_k(10).mean(),
            # Say that after doing this you don't recognize them and want to know what are their top hits,
            # Let's also take some of their most popular albums songs for reference:
            pl.col("track_name").sort_by("popularity").unique(maintain_order=True).top_k(5),
            pl.col("album_name").sort_by("popularity").unique(maintain_order=True).top_k(5),
            pl.col("track_genre").top_k_by("popularity", k=1).alias("Most popular genre"),
            # And for good measure, see how many total tracks they have
            pl.col("track_name").n_unique().alias("tracks_count"),
        )
        .sort("popularity", descending=True)
        .collect()
    )
    mo.vstack(
        [
            mo.md("Let's start by taking a look at the most popular artists"),
            # Also adjust the formatting for displaying columns that include multiple values in the same line
            most_popular_artists.with_columns(pl.col(pl.List(pl.String())).list.join("\n")),
            mo.md("Recognize any of your favourite songs? Me neither. Let's try adding a filter by genre"),
            filter_genre,
            mo.md(
                "(the code is omitted for brevity, but you can click the eye icon to see it)",
            ),
        ],
    )
    return (most_popular_artists,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So far so good - but there's been a distinct lack of visualations, so let's fix that.

        Let's start simple, just some metrics for each genre:
        """
    )
    return


@app.cell
def _(filtered_duration, pl, px):
    fig_dur_per_genre = px.scatter(
        filtered_duration.group_by("track_genre")
        .agg(
            pl.col("duration_seconds", "popularity").mean().round(2),
        )
        .sort("track_genre", descending=True),
        hover_name="track_genre",
        y="duration_seconds",
        x="popularity",
    )
    fig_dur_per_genre
    return (fig_dur_per_genre,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, why don't we play a bit with marimo's UI elements?

        We will use Dropdowns to allow for the user to select any column to use for the visualisation, and throw in some extras

        - A slider for the transparency to help understand dense clusters
        - Add a Trendline to the scatterplot (requires statsmodels)
        - Filter by some specific Genre
        """
    )
    return


@app.cell(hide_code=True)
def _(
    alpha,
    color,
    filter_genre2,
    filtered_duration,
    include_trendline,
    mo,
    pl,
    px,
    x_axis,
    y_axis,
):
    fig2 = px.scatter(
        filtered_duration.filter(
            (pl.col("track_genre") == filter_genre2.value) if filter_genre2.value is not None else True
        ),
        x=x_axis.value,
        y=y_axis.value,
        color=color.value,
        opacity=alpha.value,
        trendline="lowess" if include_trendline.value else None,
        render_mode="webgl",
        # strings on hover get fairly heavy when there are too many rows, but you can try using it after applying a few filters
        # hover_name="track_name",
        # hover_data=("artists", "album_name"),
    )
    chart2 = mo.ui.plotly(fig2)

    mo.vstack([mo.hstack([x_axis, y_axis, color, alpha, include_trendline, filter_genre2]), chart2])
    return chart2, fig2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As we have seen before, we can also use the plot as an input to select a region and look at it in more detail.

        Try selecting a region then performing some explorations of your own with the data inside of it.
        """
    )
    return


@app.cell
def _(chart2, filtered_duration, mo, pl):
    # Looking at which sort of songs were included in that region
    if len(chart2.value) == 0:
        out = mo.md("No data found in selection")
        active_columns = column_order = None
    else:
        active_columns = list(chart2.value[0].keys())
        column_order = ["track_name", *active_columns, "album_name", "artists"]
        out = filtered_duration.join(pl.DataFrame(chart2.value).unique(), on=active_columns).select(
            pl.col(column_order), pl.exclude(*column_order)
        )
    out
    return active_columns, column_order, out


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this notebook, we've focused on a few key aspects. First, it's essential to *understand* the data you're working with — this forms the foundation of any analysis. 

        Creating plots is a powerful way to identify patterns, outliers, and trends. These visualizations are not just for _presentation_; they are tools for deeper insight.

        /// NOTE
        With marimo's `interactive` UI elements, exploring different _facets_ of the data becomes seamless, allowing for dynamic analysis without altering the code.

        Keep these points in mind as you continue to work with data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Utility Functions and UI Elements""")
    return


@app.cell
def get_extremes():
    def get_extremes(selection, col, defaults_if_missing):
        "Get the minimum and maximum values for a given column within the selection"
        if selection is None or len(selection) == 0:
            print(
                f"Could not find a selected region. Using default values {defaults_if_missing} instead, try clicking and dragging in the plot to change them."
            )
            return defaults_if_missing
        else:
            return (
                min(row[col] for row in selection),
                max(row[col] for row in selection),
            )
    return (get_extremes,)


@app.cell
def _(filtered_duration, mo):
    filter_genre = mo.ui.dropdown(
        options=filtered_duration["track_genre"].unique().sort().to_list(),
        allow_select_none=True,
        value=None,
        searchable=True,
        label="Filter by Track Genre:",
    )
    return (filter_genre,)


@app.cell
def _(filtered_duration, mo):
    # Columns that make sense for the scatterplot and the corresponding UI elements
    options = [
        "duration_seconds",
        "popularity",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    x_axis = mo.ui.dropdown(options, value="energy", label="X")
    y_axis = mo.ui.dropdown(options, value="danceability", label="Y")
    color = mo.ui.dropdown(options, value="loudness", allow_select_none=True, searchable=True, label="Color column")
    alpha = mo.ui.slider(start=0.01, stop=1.0, step=0.01, value=0.1, label="Alpha", show_value=True)
    include_trendline = mo.ui.checkbox(label="Trendline")
    # We *could* reuse the same filter_genre from above, but it would cause marimo to rerun both the table and the graph whenever we change it
    filter_genre2 = mo.ui.dropdown(
        options=filtered_duration["track_genre"].unique().sort().to_list(),
        allow_select_none=True,
        value=None,
        searchable=True,
        label="Filter by Track Genre:",
    )
    return (
        alpha,
        color,
        filter_genre2,
        include_trendline,
        options,
        x_axis,
        y_axis,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Appendix : Some other examples""")
    return


@app.cell
def _(mo):
    # Components to filter for some specific song
    filter_artist = mo.ui.text(label="Artist: ")
    filter_track = mo.ui.text(label="Track name: ")
    return filter_artist, filter_track


@app.cell(disabled=True)
def _(filtered_duration, mo, pl):
    # Note that we cannot use dropdown due to the sheer number of elements being enormous:
    all_artists = filtered_duration.select(pl.col("artists").str.split(";").explode().unique().sort())["artists"].to_list()
    all_tracks = filtered_duration["track_name"].unique().sort().to_list()
    alternative_filter_artist = mo.ui.dropdown(all_artists, value=None, searchable=True)
    alternative_filter_track = mo.ui.dropdown(all_tracks, value=None, searchable=True)
    # So we just provide freeform text boxes and filter ourselfves later
    # (the "alternative_" in the name is just to avoid conflicts with the above cell,
    #  despite this being disabled marimo still requires global variables to be unique)
    return (
        all_artists,
        all_tracks,
        alternative_filter_artist,
        alternative_filter_track,
    )


@app.cell
def _(filter_artist, filter_track, filtered_duration, mo, pl):
    def score_match_text(col: pl.Expr, string: str | None) -> pl.Expr:
        if not string:
            return pl.lit(0)
        col = col.str.to_lowercase()
        string = string.casefold()
        return (
            # For a more professional use case, you might want to look into string distance functions
            # in the polars-ds package or other polars plugins
            -col.str.len_chars().cast(pl.Int32())
            + pl.when(col.str.contains(string)).then(50).otherwise(0)
            + pl.when(col.str.starts_with(string)).then(50).otherwise(0)
        )


    filtered_artist_track = (
        filtered_duration.select(
            pl.col("artists"),
            pl.col("track_name"),
            (
                score_match_text(pl.col("track_name"), filter_track.value)
                + pl.col("artists").str.split(";").list.eval(score_match_text(pl.element(), filter_artist.value)).list.sum()
            ).alias("match_score"),
            pl.col("album_name"),
            pl.col("track_genre"),
            pl.col("popularity"),
            pl.col("duration_seconds"),
        )
        .filter(pl.col("match_score") > 0)
        .sort("match_score", descending=True)
    )

    mo.vstack([mo.md("Filter a track based on its name or artist"), filter_artist, filter_track, filtered_artist_track])
    return filtered_artist_track, score_match_text


@app.cell
def _(filter_genre2, filtered_duration, mo, pl):
    # Artists combinations
    artist_combinations = (
        filtered_duration.lazy()
        .filter((pl.col("track_genre") == filter_genre2.value) if filter_genre2.value is not None else True)
        .with_columns(pl.col("artists").str.split(";"))
        .with_columns(pl.col("artists").alias("other_artist"))
        .explode("artists")
        .explode("other_artist")
        # Filter to:
        # 1) Remove an artist with themselves
        # 2) Remove duplicate combinations, otherwise we would have once row for (A, B) and one for (B, A)
        .filter(pl.col("artists") > pl.col("other_artist"))
        .group_by("artists", "other_artist")
        .len("count")
        .collect()
    )
    mo.vstack(
        [
            mo.md("Check which artists collaborate with others most often (reuses the last genre filter)"),
            filter_genre2,
            artist_combinations.sort("count", descending=True),
        ],
        align="center",
    )
    return (artist_combinations,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    import plotly.express as px
    return pl, px


if __name__ == "__main__":
    app.run()

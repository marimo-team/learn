# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "plotly[express]==6.3.0",
#     "polars==1.33.1",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dealing with Missing Data

    _by [etrotta](https://github.com/etrotta) and [Felix Najera](https://github.com/folicks)_

    This notebook covers some common problems you may face when dealing with real datasets and techniques used to solve deal with them, showcasing polars functionalities to handle missing data.

    First we provide an overview of the methods available in polars, then we walk through a mini case study with real world data showing how to use it, and at last we provide some additional information in the 'Bonus Content' section.
    You can navigate to skip around to each header using the menu on the right side
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Methods for working with Nulls

    We'll be using the following DataFrame to show the most important methods:
    """)
    return


@app.cell(hide_code=True)
def _(pl):
    df = pl.DataFrame(
        [
            {"species": "Dog", "name": "Millie", "height": None, "age": 4},
            {"species": "Dog", "name": "Wally", "height": 60, "age": None},
            {"species": "Dog", "name": None, "height": 50, "age": 12},
            {"species": "Cat", "name": "Mini", "height": 15, "age": None},
            {"species": "Cat", "name": None, "height": 25, "age": 6},
            {"species": "Cat", "name": "Kazusa", "height": None, "age": 16},
        ]
    )
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Counting nulls

    A simple yet convenient aggregation
    """)
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dropping Nulls

    The simplest way of dealing with null values is throwing them away, but that is not always a good idea.
    """)
    return


@app.cell
def _(df):
    df.drop_nulls()
    return


@app.cell
def _(df):
    df.drop_nulls(subset="name")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filtering null values

    To filter in polars, you'll typically use `df.filter(expression)` or `df.remove(expression)` methods.

    Filter will only keep rows in which the expression evaluates to True.
    It will remove not only rows in which it evaluates to False, but also those in which the expression evaluates to None.

    Remove will only remove rows in which the expression evaluates to True.
    It will keep rows in which it evaluates to None.
    """)
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("age") > 10)
    return


@app.cell
def _(df, pl):
    df.remove(pl.col("age") < 10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You may also be tempted to use `== None` or `!= None`, but operators in polars will generally propagate null values.

    You can use `.eq_missing()` or `.ne_missing()` methods if you want to be strict about it, but there are also `.is_null()` and `.is_not_null()` methods you can use.
    """)
    return


@app.cell
def _(df, pl):
    df.select(
        "name",
        (pl.col("name") == None).alias("Name equals None"),
        (pl.col("name") == "Mini").alias("Name equals Mini"),
        (pl.col("name").eq_missing("Mini")).alias("Name eq_missing Mini"),
        (pl.col("name").is_null()).alias("Name is null"),
        (pl.col("name").is_not_null()).alias("Name is not null"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filling Null values

    You can also fill in the values with constants, calculations or by consulting external data sources.

    Be careful not to treat estimated or guessed values as if they a ground truth however, otherwise you may end up making conclusions about a reality that does not exists.

    As an exercise, let's guess some values to fill in nulls, then try giving names to the animals with `null` by editing the cells
    """)
    return


@app.cell
def _(df, mo, pl):
    guesstimates = df.with_columns(
        pl.col("height").fill_null(pl.col("height").mean().over("species")),
        pl.col("age").fill_null(0),
    )
    guesstimates = mo.ui.data_editor(
        guesstimates,
        editable_columns=["name"],
    )
    guesstimates
    return (guesstimates,)


@app.cell
def _(guesstimates):
    guesstimates.value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TL;DR

    Before we head into the mini case study, a brief review of what we have covered:

    - use `df.null_counts()` or `expr.is_null()` to count and identify missing values
    - you could just drop rows with values missing in any columns or a subset of them with `df.drop_nulls()`, but for most cases you'll want to be more careful about it
    - take into consideration whenever you want to preserve null values or remove them when choosing between `df.filter()` or `df.remove()`
    - if you don't want to propagate null values, use `_missing` variations of methods such as `eq` vs `eq_missing`
    - you may want to fill in missing values based on calculations via `fill_null`, join and coalesce based on other datasets, or manually edit the data based on external documents

    You can also refer to the polars [User Guide](https://docs.pola.rs/user-guide/expressions/missing-data/) more more information.

    Whichever approach you take, remember to document how you handled it!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mini Case Study

    We will be using a dataset from `alertario` about the weather in Rio de Janeiro, originally available in Google Big Query under `datario.clima_pluviometro`. What you need to know about it:

    - Contains multiple stations covering the Municipality of Rio de Janeiro
    - Measures the precipitation as millimeters, with a granularity of 15 minutes
    - We filtered to only include data about 2020, 2021 and 2022
    """)
    return


@app.cell
def _(px, stations):
    px.scatter_map(stations, lat="lat", lon="lon", text="name")
    return


@app.cell(disabled=True, hide_code=True)
def _(pl, px, stations):
    # In case `scatter_map` does not works for you:
    _fig = px.scatter_geo(stations, lat="lat", lon="lon", hover_name="name")

    _min_lat = stations.select(pl.col("lat").min()).item()
    _max_lat = stations.select(pl.col("lat").max()).item()
    _min_lon = stations.select(pl.col("lon").min()).item()
    _max_lon = stations.select(pl.col("lon").max()).item()

    _fig.update_geos(
        lataxis_range=[_min_lat - 0.2, _max_lat + 0.2],
        lonaxis_range=[_min_lon - 0.2, _max_lon + 0.2],
        resolution=50,
        showocean=True,
        oceancolor="Lightblue",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stations

    First, let's take a look at some of the stations. Notice how

    - Some stations have been deactivated, so there won't be any data about them (in fact, we don't even know their coordinates)
    - There are some columns that do not even contain data at all!

    We will remove the empty columns and remove rows without coordinates
    """)
    return


@app.cell(hide_code=True)
def _(dirty_stations, mo, pl):
    # If you were working on this yourself, you may want to briefly at *all* of them, but for practical purposes I am taking a slice for the displayed output, as otherwise it would take too much screen space.
    # mo.ui.table(dirty_stations, pagination=False)

    mo.vstack(
        [
            mo.md("Before (head and tail sample):"),
            pl.concat([dirty_stations.head(3), dirty_stations.tail(3)], how="vertical"),
        ]
    )
    return


@app.cell
def _(dirty_stations, mo, pl):
    stations = dirty_stations.drop_nulls(subset=("lat", "lon")).drop(pl.col(r"^operation_(start|end)_date$"))
    mo.vstack([mo.md("After (full dataframe):"), stations])
    return (stations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Precipitation
    Now, let's move on to the Precipitation data.

    ## Part 1 - Null Values

    First of all, let's check for null values:
    """)
    return


@app.cell
def _(dirty_weather, pl):
    rain = pl.col("accumulated_rain_15_minutes")  # Create an alias since we'll use that column a lot

    dirty_weather.filter(rain.is_null())
    return (rain,)


@app.cell(hide_code=True)
def _(dirty_weather, mo, rain):
    _missing_count = dirty_weather.select(rain.is_null().sum()).item()

    mo.md(
        f"As you can see, there are {_missing_count:,} rows missing the accumulated rain for a period.\n\nThat could be caused by sensor malfunctions, maintenance, bobby tables or a myriad of other reasons. While it may be a small percentage of the data ({_missing_count / len(dirty_weather):.3%}), it is still important to take it in consideration, one way or the other."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### First option to fixing it: Dropping data.

    We could just remove those rows like we did for the stations, which may be a passable solution for some problems, but is not always the best idea.
    ```py
    dirty_weather.drop_nulls()
    ```

    ### Second option to fixing it: Interpolation

    Instead of removing these rows, we can use some heuritics to guess values that make sense for them. Remember that this adds a degree of uncertainty to the final results, so you should disclose how you are treating missing values if you draw any conclusions based on such guesses.
    ```py
    dirty_weather.with_columns(rain.fill_null(strategy="forward")),
    ```

    When doing so, which strategy may make sense for your data varies greatly. In some cases you'll want to use the mean to maintain it centered around the same distribution, while in other cases you'll want to zero it to avoid modifying the total, or fill forward/backward to keep it mostly continuous.

    ### Last option to fixing it: Acquire the correct values from elsewhere.

    Like manually adding names to the animals in the introduction, but you could try finding approximate values from another dataset or in some cases manually input the correct values.

    ### However

    Let's investigate a bit more before deciding on following with either approach.
    For example, is our current data even complete, or are we already missing some rows beyond those with null values?
    """)
    return


@app.cell
def _(dirty_weather, pl):
    seen_counts = dirty_weather.group_by(pl.col("datetime").dt.time(), "station").len()

    # Fun fact: a single row has its time set to `23:55`.
    # It should not be present in this dataset, but found its way into the official Google Big Query table somehow.
    seen_counts = seen_counts.filter(pl.col("len") > 1)
    # You may want to treat it as a bug or outlier and remove it from dirty_weather, but we won't dive into cleaning such in this notebook

    # seen_counts.sort("station", "datetime").select("station", "datetime", "len")
    seen_counts.sort("len").select("station", "datetime", "len")
    return


@app.cell
def _(pl):
    expected_range = pl.datetime_range(
        pl.lit("2020-01-01T00:00:00").str.to_datetime(time_zone="America/Sao_Paulo"),
        pl.lit("2022-12-31T23:45:00").str.to_datetime(time_zone="America/Sao_Paulo"),
        "15m",
    )

    pl.select(expected_range).group_by(pl.col.literal.dt.time()).len().sort("literal")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 - Missing Rows

    We can see that we expected there to be 1096 rows for each hour for each station (from the start of 2020 to the end of 2022) , but in reality we see between 1077 and 1096 rows.

    That difference could be caused by the same factors as null values, or even by someone dropping null values along the way, but for the purposes of this notebook let's say that we want to have values for each combination with no exceptions, so we'll have to make reasonable assumptions to interpolate and extrapolate them.

    ### Upsampling

    Given that we are working with time series data, we will [upsample](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.upsample.html) the data, but you could also create a DataFrame containing all expected rows then use `join(how="...")`

    However, that will give us _even more_ null values, so we will want to fill them in afterwards. For this case, we will just use a forward fill followed by a backwards fill.
    """)
    return


@app.cell
def _(dirty_weather, mo, pl, rain):
    _hollow_weather = dirty_weather.sort("station", "datetime").upsample("datetime", every="15m", group_by="station")
    weather = _hollow_weather.fill_null(strategy="forward").fill_null(strategy="backward")

    mo.vstack(
        [
            mo.ui.table(
                label="Null counts at each step",
                data=pl.concat(
                    [
                        dirty_weather.null_count().select(
                            pl.lit("Before upsampling").alias("label"), rain, "station", "datetime"
                        ),
                        _hollow_weather.null_count().select(
                            pl.lit("After upsampling").alias("label"), rain, "station", "datetime"
                        ),
                        weather.null_count().select(pl.lit("After filling").alias("label"), rain, "station", "datetime"),
                    ]
                ),
            ),
            mo.md("Data after upsampling and filling in nulls:"),
            weather,
        ]
    )
    return (weather,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we finally have a clean dataset, let's play around with it a little.

    ### Example App

    Let's display the amount of precipitation each station measured within a timeframe, aggregated to a lower granularity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    filters = (
        mo.md(
            """Filters for the example

        Year: {year}
        Days of the year: {day}
        Hours of each day: {hour}
        Aggregation granularity: {interval}
        """
        )
        .batch(
            year=mo.ui.dropdown([2020, 2021, 2022], value=2022),
            day=mo.ui.range_slider(1, 365, show_value=True, full_width=True, value=[87, 94]),
            hour=mo.ui.range_slider(0, 24, 0.25, show_value=True, full_width=True),
            interval=mo.ui.dropdown(["15m", "30m", "1h", "2h", "4h", "6h", "1d", "7d", "30d"], value="4h"),
        )
        .form()
    )

    # Note: You could use `mo.ui.date_range` instead, but I just don't like it myself
    # mo.ui.date_range(start="2020-01-01", stop="2022-12-31", value=["2022-03-28", "2022-04-03"], label="Display range")

    filters
    return (filters,)


@app.cell
def _(filters, mo, pl, rain, stations, weather):
    mo.stop(filters.value is None)

    _range_seconds = map(lambda hour: hour * 3600, filters.value["hour"])
    _df_seconds = pl.col("datetime").dt.hour().cast(pl.Float64()).mul(3600) + pl.col("datetime").dt.minute().cast(
        pl.Float64()
    ).mul(60)

    animation_data = (
        weather.lazy()
        .filter(
            pl.col("datetime").dt.year() == filters.value["year"],
            pl.col("datetime").dt.ordinal_day().is_between(*filters.value["day"]),
            _df_seconds.is_between(*_range_seconds),
        )
        .group_by_dynamic("datetime", group_by="station", every=filters.value["interval"])
        .agg(rain.sum().alias("precipitation"))
        .remove(pl.col("precipitation").eq(0).all().over("station"))
        .join(stations.lazy(), on="station")
        .select("name", "lat", "lon", "precipitation", "datetime")
        .collect()
    )
    return (animation_data,)


@app.cell
def _(animation_data, pl, px):
    _fig = px.scatter_geo(
        animation_data.with_columns(avg_precipitation=pl.col("precipitation").mean()),
        lat="lat",
        lon="lon",
        hover_name="name",
        animation_group="name",
        animation_frame="datetime",
        size="avg_precipitation",
        color="precipitation",
        color_continuous_scale="PuBu",
        range_color=[0, animation_data.select(pl.col("precipitation").max()).item()],
    )

    _min_lat = animation_data.select(pl.col("lat").min()).item()
    _max_lat = animation_data.select(pl.col("lat").max()).item()
    _min_lon = animation_data.select(pl.col("lon").min()).item()
    _max_lon = animation_data.select(pl.col("lon").max()).item()

    _fig.update_geos(
        lataxis_range=[_min_lat - 0.2, _max_lat + 0.2],
        lonaxis_range=[_min_lon - 0.2, _max_lon + 0.2],
        resolution=50,
        showocean=True,
        oceancolor="Lightblue",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we were missing some rows, we would have circles popping in and out of existence instead of a smooth animation!

    In many scenarios, missing data can also lead to wrong results overall, for example if we were to estimate the total amount of rainfall during the observed period:
    """)
    return


@app.cell
def _(dirty_weather, mo, rain, weather):
    old_estimate = dirty_weather.select(rain.sum()).item()
    new_estimate = weather.select(rain.sum()).item()
    # Note: The aggregation used to calculate these variables (taking a sum across all stations) is not very meaningful, but the relative difference between them scales across many potentially useful aggregations

    mo.md(f"Our estimates may change by roughly {(new_estimate - old_estimate) / old_estimate:.2%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which is still a relatively small difference, but every drop counts when you are dealing with the weather.

    For datasets with a higher share of missing values, that difference can get much higher.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bonus Content

    ## Appendix A: Missing Time Zones

    The original dataset contained naive datetimes instead of timezone-aware, but we can infer whenever it refers to UTC time or local time (for this case, -03:00 UTC) based on the measurements.

    For example, we can select one specific interval during which we know that rained a lot, or graph the average amount of precipitation for each hour of the day, then compare the data timestamps with a ground truth.
    """)
    return


@app.cell(hide_code=True)
def _(dirty_weather_naive, mo):
    mo.vstack(
        [
            mo.md("Original data example:"),
            dirty_weather_naive.head(3),
        ]
    )
    return


@app.cell
def _(dirty_weather_naive, pl, px, rain):
    naive_downfall_per_hour = (
        dirty_weather_naive.group_by(pl.col("datetime").dt.hour().alias("hour"))
        .agg(rain.sum().alias("accumulated_rain"))
        .with_columns(pl.col("accumulated_rain").truediv(pl.col("accumulated_rain").sum()).mul(100))
    )
    px.bar(
        naive_downfall_per_hour.sort("hour"),
        x="hour",
        y="accumulated_rain",
        title="Distribution of precipitation per hour (%), using the naive datetime",
    )
    return


@app.cell
def _(dirty_weather_naive, pl, rain, stations):
    naive_top_rain_events = (
        dirty_weather_naive.lazy()
        # If you wanted to filter the dates and locate a specific event:
        # .filter(pl.col("datetime").is_between(pl.lit("2022-03-01").str.to_datetime(), pl.lit("2022-05-01").str.to_datetime()))
        .sort("station", "datetime")
        .group_by_dynamic("datetime", every="1h", offset="30m", group_by="station")
        .agg(rain.sum())
        .join(stations.lazy(), on="station")
        .sort(rain, descending=True)
        .select(
            "name",
            pl.col("datetime").alias("window_start"),
            (pl.col("datetime") + pl.duration(hours=1)).alias("window_end"),
            rain.alias("accumulated rain"),
        )
        .head(50)
        .collect()
    )
    naive_top_rain_events
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By externally researching the expected distribution and looking up some of the extreme weather events, we can come to a conclusion about whenever it is aligned with the local time or with UTC.

    In this case, the distribution matches the normal weather for this region and we can see that the hours with the most precipitation match those of historical events, so it is safe to say it is using local time (equivalent to the Americas/São Paulo time zone).
    """)
    return


@app.cell
def _(dirty_weather_naive, pl):
    dirty_weather = dirty_weather_naive.with_columns(pl.col("datetime").dt.replace_time_zone("America/Sao_Paulo"))

    dirty_weather.head(3)
    return (dirty_weather,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix B: Not a Number

    While some other tools without proper support for missing values may use `NaN` as a way to indicate a value is missing, in polars it is treated exclusively as a float value, much like `0.0`, `1.0` or `infinity`.

    You can use `.fill_null(float('nan'))` if you need to convert floats to a format such tools accept, or use `.fill_nan(None)` if you are importing data from them, assuming that there are no values which really are supposed to be the float NaN.

    Remember that many calculations can result in NaN, for example dividing by zero:
    """)
    return


@app.cell
def _(dirty_weather, pl, rain):
    day_perc = dirty_weather.select(
        "datetime",
        (rain / rain.sum().over("station", pl.col("datetime").dt.date())).alias("percentage_of_day_precipitation"),
    )
    perc_col = pl.col("percentage_of_day_precipitation")

    day_perc
    return day_perc, perc_col


@app.cell(hide_code=True)
def _(day_perc, mo, perc_col):
    mo.md(
        f"""
    It is null for {day_perc.select(perc_col.is_null().mean()).item():.4%} of the rows, but is NaN for {day_perc.select(perc_col.is_nan().mean()).item():.4%} of them.
    If we use the cleaned weather dataframe to calculate it instead of the dirty_weather, we will have no nulls, but note how for this calculation we can end up with both, with each having a different meaning.

    In this case it makes sense to fill in NaNs as 0 to indicate there was no rain during that period, but treating the nulls the same could lead to a different interpretation of the data, so remember to handle NaNs and nulls separately.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix C: Everything else

    As long as this Notebook is, it cannot reasonably cover ***everything*** that may have to deal with missing values, as that is literally everything that may have to deal with data.

    This section very briefly covers some other features not mentioned above
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Missing values in Aggregations

    Many aggregations methods will ignore/skip missing values, while others take them into consideration.

    Always check the documentation of the method you're using, much of the time docstrings will explain their behaviour.
    """)
    return


@app.cell
def _(df, pl):
    df.group_by("species").agg(
        pl.col("height").len().alias("len"),
        pl.col("height").count().alias("count"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Missing values in Joins

    By default null values will never produce matches using [join](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html), but you can specify `nulls_equal=True` to join Null values with each other.
    """)
    return


@app.cell(hide_code=True)
def _(pl):
    age_groups = pl.DataFrame(
        [
            {"age": None, "stage": "Unknown"},
            {"age": [0, 1], "stage": "Baby"},
            {"age": [2, 3, 4, 5, 6, 7, 8, 9, 10], "stage": "Adult"},
            {"age": [11, 12, 13, 14], "stage": "Senior"},
            {"age": [15, 16, 17, 18, 19, 20], "stage": "Geriatric"},
        ]
    )
    age_groups
    return (age_groups,)


@app.cell
def _(age_groups, df):
    df.join(age_groups.explode("age"), on="age")
    return


@app.cell
def _(age_groups, df):
    df.join(age_groups.explode("age"), on="age", nulls_equal=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utilities

    Loading data and imports
    """)
    return


@app.cell
def _(pl):
    raw_stations = pl.scan_csv("hf://datasets/etrotta/weather-alertario/datario_alertario_stations.csv")
    raw_weather = pl.scan_csv("hf://datasets/etrotta/weather-alertario/datario_alertario_weather_2020_to_2022.csv")
    return raw_stations, raw_weather


@app.cell
def _(pl, raw_stations):
    dirty_stations = raw_stations.select(
        pl.col("id_estacao").alias("station"),
        pl.col("estacao").alias("name"),
        pl.col("latitude").alias("lat"),
        pl.col("longitude").alias("lon"),
        pl.col("cota").alias("altitude"),
        pl.col("situacao").alias("situation"),
        pl.col("endereco").alias("address"),
        pl.col("data_inicio_operacao").alias("operation_start_date"),
        pl.col("data_fim_operacao").alias("operation_end_date"),
    ).collect()
    return (dirty_stations,)


@app.cell
def _(pl, raw_weather):
    dirty_weather_naive = raw_weather.select(
        pl.col("id_estacao").alias("station"),
        pl.col("acumulado_chuva_15_min").alias("accumulated_rain_15_minutes"),
        pl.concat_str("data_particao", pl.lit("T"), "horario").str.to_datetime(time_zone=None).alias("datetime"),
    ).collect()
    return (dirty_weather_naive,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _():
    import plotly.express as px
    return (px,)


if __name__ == "__main__":
    app.run()

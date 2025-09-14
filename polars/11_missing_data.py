# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "plotly[express]==6.3.0",
#     "polars==1.33.1",
# ]
# ///

import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Dealing with Missing Data

    _by [etrotta](https://github.com/etrotta)_

    This notebook covers some common problems you may face when dealing with real datasets and techniques used to solve deal with them, providing an overview of polars functionalities to handle missing data.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We will be using a dataset about the weather in Rio de Janeiro, originally available in Google Big Query under `datario.clima_pluviometro`. What you need to know about it: 

    - Contains multiple stations covering the Municipality of Rio de Janeiro
    - Measures the precipitation as milimeters, with a granularity of 15 minutes
    - We filtered to only include data about 2020, 2021 and 2022
    """
    )
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
    mo.md(
        r"""
    # Stations

    First, let's take a look at some of the stations. Notice how

    - Some stations have been deactivated, so there won't be any data about them (in fact, we don't even know their coordinates)
    - There are some columns that do not even contain data at all!

    We will remove the empty columns and remove rows without coordinates
    """
    )
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
    mo.md(
        r"""
    # Precipitation
    Now, let's move on to the Precipitation data.

    ## Part 1 - Null Values

    First of all, let's check for null values:
    """
    )
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
        f"As you can see, there are {_missing_count:,} rows missing the accumulated rain for a period.\n\nThat could be cause due to sensor malfunctions, maintenance, bobby tables or a myriad of other reasons. While it may be a small percentage of the data ({_missing_count / len(dirty_weather):.3%}), it is still important to take it in consideration, one way or the other."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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

    We will not explore this option in this notebook, but you could try finding approximate values from another dataset or in some cases manually input the correct values.

    ### However

    Let's investigate a bit more before deciding on following with either approach.
    For example, is our current data even complete, or are we already missing some rows beyond those with null values?
    """
    )
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
    mo.md(
        r"""
    ## Part 2 - Missing Rows

    We can see that we expected there to be 1096 rows for each hour for each station (from the start of 2020 to the end of 2022) , but in reality we see between 1077 and 1096 rows.

    That difference could be caused by the same factors as null values, or even by someone dropping null values along the way, but for the purposes of this notebook let's say that we want to have values for each combination with no exceptions, so we'll have to make reasonable assumptions to interpolate and extrapolate them.

    Given that we are working with time series data, we will [upsample](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.upsample.html) the data, but you could also create a DataFrame containing all expected rows then use `join(how="...")`

    However, that will give us _even more_ null values, so we will want to fill them in afterwards. For this case, we will just use a forward fill followed by a backwards fill.
    """
    )
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
    mo.md(r"""Now that we finally have a clean dataset, let's play around with it a little""")
    return


@app.cell(hide_code=True)
def _(mo):
    year_picker = mo.ui.dropdown(options=[2020, 2021, 2022], value=2022, label="Year")
    day_slider = mo.ui.range_slider(1, 365, show_value=True, label="Day of the year", full_width=True, value=[87, 94])
    hour_slider = mo.ui.range_slider(0, 24, 0.25, show_value=True, label="Hour of the day", full_width=True)
    interval = mo.ui.dropdown(
        options=["15m", "30m", "1h", "2h", "4h", "6h", "1d"], value="4h", label="Aggregation Granularity"
    )

    mo.vstack(
        [
            year_picker,
            day_slider,
            hour_slider,
            interval,
        ]
    )
    return day_slider, hour_slider, interval, year_picker


@app.cell
def _(
    day_slider,
    hour_slider,
    interval,
    pl,
    rain,
    stations,
    weather,
    year_picker,
):
    _range_seconds = map(lambda hour: hour * 3600, hour_slider.value)
    _df_seconds = pl.col("datetime").dt.hour() + pl.col("datetime").dt.minute().mul(60)

    animation_data = (
        weather.lazy()
        .filter(
            pl.col("datetime").dt.year() == year_picker.value,
            pl.col("datetime").dt.ordinal_day().is_between(*day_slider.value),
            _df_seconds.is_between(*_range_seconds),
        )
        .group_by_dynamic("datetime", group_by="station", every=interval.value)
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
    mo.md(
        r"""
    If we were missing some rows, we would have circles popping in and out of existince instead of a smooth animation!

    In many scenarios, missing data can also lead to wrong results overall, for example if we were to estimate the total amount of rainfall during the observed period:
    """
    )
    return


@app.cell
def _(dirty_weather, mo, rain, weather):
    old_estimate = dirty_weather.select(rain.sum()).item()
    new_estimate = weather.select(rain.sum()).item()
    # Note: The aggregation used to calculate these variables (taking a sum across all stations) is not very meaningful, but the relative diference between them scales across many potentially useful aggregations

    mo.md(f"Our estimates may change by roughly {(new_estimate - old_estimate) / old_estimate:.2%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Which is still a relatively small difference, but every drop counts when you are dealing with the weather.

    For datasets with a higher share of missing values, that difference can get much higher.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bonus Content

    ### Appendix A: Missing Time Zones

    The original dataset contained naive datetimes instead of timezone-aware, but we can infer whenever it refers to UTC time or local time (for this case, -03:00 UTC) based on the measurements.

    For example, we can select one specific interval during which we know rained a lot, or graph the average amount of precipitation for each hour of the day, then compare the data timestamps with a ground truth.
    """
    )
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
    mo.md(
        r"""
    By externally researching the expected distribution and looking up some of the extreme weather events, we can come to a conclusion about whenever it is aligned with the local time or with UTC.

    In this case, the distribution matches the normal weather for this region and we can see that the hours with the most precipitation match those of historical events, so it is safe to say it is using Americas/São Paulo time zone.
    """
    )
    return


@app.cell
def _(dirty_weather_naive, pl):
    dirty_weather = dirty_weather_naive.with_columns(pl.col("datetime").dt.replace_time_zone("America/Sao_Paulo"))

    # Also get rid of some of the other variables to economize memory
    # del raw_weather
    # del dirty_weather_naive

    dirty_weather.head(3)
    return (dirty_weather,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Utilities

    Loading data and imports
    """
    )
    return


@app.cell
def _(pl):
    raw_stations = pl.read_csv("/mnt/c/Users/Etrot/Downloads/datario_alertario_stations.csv")
    raw_weather = pl.read_csv("/mnt/c/Users/Etrot/Downloads/datario_alertario_weather_2020_to_2022.csv")
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
    )
    return (dirty_stations,)


@app.cell
def _(pl, raw_weather):
    dirty_weather_naive = raw_weather.select(
        pl.col("id_estacao").alias("station"),
        pl.col("acumulado_chuva_15_min").alias("accumulated_rain_15_minutes"),
        pl.concat_str("data_particao", pl.lit("T"), "horario").str.to_datetime(time_zone=None).alias("datetime"),
    )
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

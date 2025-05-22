# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Reactive Histograms & Bar Charts

    _By [Aarshia Gupta](https://github.com/aarshiagupta)._

    This interactive tutorial explores **data visualization** using `plotly.express` and **Marimo widgets**. It focuses on understanding distributions and category breakdowns through dynamic **histograms and bar charts**.

    ---

    We'll be working with the **Palmer Penguins** dataset — a rich, tidy alternative to the Iris dataset — perfect for practicing **exploratory data analysis (EDA)**. 
    It includes measurements for three penguin species observed on islands in Antarctica.

    **🔢 Numeric Features**

    - `flipper_length_mm` — wing length
    - `bill_length_mm`, `bill_depth_mm` — beak measurements
    - `body_mass_g` — body weight

    **🔣 Categorical Features**

    - `species` — penguin type
    - `island` — location observed
    - `sex` — biological sex

    With this notebook, you can:

    - 🔍 Scroll, filter, and interact with a live preview of the dataset
    - 📊 Create and explore **histograms** of numeric variables
    - 🟦 Compare **bar charts** across groupings (e.g. species × sex)
    - 🧩 Segment categories using dropdowns and sliders
    - 🧠 Answer EDA questions like:
      > *Which species is most common?*  
      > *Does body mass vary by sex or island?*  
      > *Are distributions unimodal or skewed?*

    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##🐧 Dataset Preview

    This is a snapshot of the **Palmer Penguins dataset** — a clean, fun, and rich dataset to explore species, size, and sex distributions.

    - Numerical: `flipper_length_mm`, `bill_length_mm`, `bill_depth_mm`, `body_mass_g`
    - Categorical: `species`, `island`, `sex`

    You can:

    - scroll across the columns,
    - filter by clicking on column histograms,
    - and download the full CSV.
    """
    )
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 Exploring Penguin Data with Histograms
    Histograms help us understand how numerical variables like flipper length or body mass are distributed across penguins.

    You can try:

    - Switching to body_mass_g to check the weight distribution across species.
    - Increasing the number of bins to reveal finer patterns or decreasing them for a simplified overview.
  
    This section helps you explore:

    - Outliers: Are there unusual values far from the rest?
    - Skewness: Is the data leaning left or right?
    - Modality: Is the distribution single-peaked (unimodal) or does it have multiple peaks (multimodal)?
    """
    )
    return


@app.cell
def _(df, mo):
    num_cols = df.select_dtypes("number").columns.tolist()

    hist_dropdown = mo.ui.dropdown(num_cols, value="flipper_length_mm", label="Histogram Column")
    bin_slider = mo.ui.slider(start=5, stop=100, value=20, step=5, label="Number of Bins")
    return bin_slider, hist_dropdown


@app.cell
def _(bin_slider, df, hist_dropdown, mo, px):
    mo.vstack([
        mo.md("### 📉 Select and Explore a Numeric Column"),
        mo.hstack([hist_dropdown, bin_slider]),
        mo.ui.plotly(
            px.histogram(
                df,
                x=hist_dropdown.value,
                nbins=bin_slider.value,
                title=f"Histogram: {hist_dropdown.value}",
                color_discrete_sequence=["#636EFA"]
            ).update_layout(title_x=0.5)
        )
    ])

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 🧮 Exploring Categorical Features with Bar Charts

    Bar charts show how often each **category** appears in your data. For instance:

    - Choosing `species` reveals the frequency of each penguin type.
    - Selecting `island` helps identify where penguins were most commonly observed.
    - Selecting `sex` shows the gender split — which can be useful if you’re studying biological differences.

    Try switching categories with the dropdown to observe patterns, imbalances, or missing values.

    """
    )
    return


@app.cell
def _(df, mo):
    # Get categorical columns
    cat_cols_all = df.select_dtypes("object").columns.tolist()
    cat_cols_grouped = [col for col in cat_cols_all if col != "species"]


    # Create dropdown to select category column
    bar_dropdown = mo.ui.dropdown(cat_cols_all, value="species", label="Bar Chart Column")

    return bar_dropdown, cat_cols_grouped


@app.cell
def _(bar_dropdown, df, mo, px):
    mo.vstack([
        mo.md("### 📊 Select a Categorical Column"),
        bar_dropdown,
        mo.ui.plotly(
            px.bar(
                df[bar_dropdown.value].value_counts()
                  .reset_index()
                  .rename(columns={"index": bar_dropdown.value}),
                x="count",
                y=bar_dropdown.value,
                orientation="h",
                title=f"Bar Chart: {bar_dropdown.value}",
                color_discrete_sequence=["#EF553B"]
            ).update_layout(title_x=0.5)
        )
    ])

    return


@app.cell
def _(cat_cols_grouped, mo):
    group_x_dropdown = mo.ui.dropdown(cat_cols_grouped, value="island", label="Main Category (x-axis)")
    group_color_dropdown = mo.ui.dropdown(cat_cols_grouped, value="sex", label="Group by (color)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 📚 Segmenting Bar Charts for Deeper Insight

    Sometimes a simple bar chart isn’t enough — we want to **compare categories within categories**.

    This grouped bar chart lets us:


    - Split each bar into **subgroups** (e.g., `sex`)
    - Click on the **legend** to isolate groups 🔍
    - Hover to explore **exact counts**

    > Try filtering by one group (e.g., female only). Which species has the most skewed sex ratio?

    """
    )
    return


@app.cell
def _(df, mo, px):
    grouped_df = (
        df.groupby(["species", "sex"], as_index=False)
          .agg(Count=("species", "size"))
    )

    fig = px.bar(
        grouped_df,
        x="species",
        y="Count",
        color="sex",
        barmode="group",
        color_discrete_sequence=["#4C78A8", "#F58518"]
    )

    fig.update_layout(
        title="Grouped Bar Chart: species by sex",
        title_x=0.5
    )

    mo.ui.plotly(fig)

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 📊 Comparing Multiple Features: Side-by-Side Charts

    This section lets you explore **how two features relate**, interactively.

    Try:

    - Changing the **grouping** on the bar chart (e.g., by `sex` or `island`)
    - Picking a **numerical variable** for the histogram (e.g., `body_mass_g`, `flipper_length_mm`)
    - Adjusting **bin size** to smooth or sharpen the histogram

    These charts let you compare:

    - Categorical distribution patterns (e.g., gender spread within species)
    - Numeric data distributions across the dataset

    Use the dropdowns above to explore trends across variables!

    """
    )
    return


@app.cell
def _(df, mo):
    cat_cols = df.select_dtypes("object").columns.drop("species").tolist()  # exclude species to avoid reset_index errors
    numeric_cols = df.select_dtypes("number").columns.tolist()

    # UI elements
    bar_group_dropdown = mo.ui.dropdown(cat_cols, value="sex", label="Bar Chart Group By")
    hist_num_dropdown = mo.ui.dropdown(numeric_cols, value="body_mass_g", label="Histogram Variable")
    bins_slider = mo.ui.slider(start=5, stop=50, value=20, step=5, label="Number of Bins")

    return bar_group_dropdown, bins_slider, hist_num_dropdown


@app.cell
def _(bar_group_dropdown, bins_slider, df, hist_num_dropdown, mo, px):
    grp_df = (
        df.groupby(["species", bar_group_dropdown.value], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )

    bar_fig = px.bar(
        grp_df,
        x="species",
        y="Count",
        color=bar_group_dropdown.value,
        barmode="group",
        color_discrete_sequence=["#4C78A8", "#F58518", "#72B7B2"]
    ).update_layout(title=f"species × {bar_group_dropdown.value}", title_x=0.5)

    hist_fig = px.histogram(
        df,
        x=hist_num_dropdown.value,
        nbins=bins_slider.value,
        color_discrete_sequence=["#A5C8E1"]
    ).update_layout(title=f"Distribution of {hist_num_dropdown.value}", title_x=0.5)

    mo.vstack([
        mo.hstack([bar_group_dropdown, hist_num_dropdown, bins_slider]),
        mo.hstack([mo.ui.plotly(bar_fig), mo.ui.plotly(hist_fig)])
    ])

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## ✨ Wrapping Up

    That’s a quick dive into the Palmer Penguins dataset using interactive bar charts and histograms. 
    We compared group counts, explored numeric distributions, and used a few widgets to make it all a bit more dynamic.

    There’s still plenty more to try—like filtering specific species, layering distributions, or adding trendlines—but this gives a solid starting point for exploratory data viz.

    Thanks for scrolling through 🐧

    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px

    app = mo.App()

    df = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv").dropna()
    return df, mo, px


if __name__ == "__main__":
    app.run()

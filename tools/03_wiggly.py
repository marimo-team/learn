# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "altair",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "wigglystuff==0.3.1",
# ]
# ///
import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # wigglystuff Widgets

    This notebook demonstrates four widgets from the [wigglystuff](https://github.com/koaning/wigglystuff) package:
    `Slider2D`, `Matrix`, `HoverZoom`, and `TextCompare`.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wigglystuff import Slider2D, Matrix, HoverZoom, TextCompare
    return HoverZoom, Matrix, Slider2D, TextCompare, alt, mo, np, pd, plt


# ---------------------------------------------------------------------------
# Slider2D
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Slider2D

    A two-dimensional slider that lets you pick an (x, y) point inside a bounded region.
    """)
    return


@app.cell
def _(Slider2D, mo):
    slider2d = mo.ui.anywidget(
        Slider2D(
            width=320,
            height=320,
            x_bounds=(-2.0, 2.0),
            y_bounds=(-1.0, 1.5),
        )
    )
    slider2d
    return (slider2d,)


@app.cell
def _(mo, slider2d):
    mo.callout(
        f"x = {slider2d.x:.3f}, y = {slider2d.y:.3f};  bounds {slider2d.x_bounds} / {slider2d.y_bounds}"
    )
    return


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix

    An editable matrix widget. The demo below applies a 3x2 transformation matrix to
    1000 random RGB colours and plots the result in 2D — a manual PCA explorer.
    """)
    return


@app.cell
def _(Matrix, mo, np, pd):
    pca_mat = mo.ui.anywidget(Matrix(np.random.normal(0, 1, size=(3, 2)), step=0.1))
    rgb_mat = np.random.randint(0, 255, size=(1000, 3))
    color = ["#{0:02x}{1:02x}{2:02x}".format(r, g, b) for r, g, b in rgb_mat]
    rgb_df = pd.DataFrame(
        {"r": rgb_mat[:, 0], "g": rgb_mat[:, 1], "b": rgb_mat[:, 2], "color": color}
    )
    return color, pca_mat, rgb_df, rgb_mat


@app.cell
def _(alt, color, mo, pca_mat, pd, rgb_mat):
    X_tfm = rgb_mat @ pca_mat.matrix
    df_pca = pd.DataFrame({"x": X_tfm[:, 0], "y": X_tfm[:, 1], "c": color})
    pca_chart = (
        alt.Chart(df_pca)
        .mark_point()
        .encode(x="x", y="y", color=alt.Color("c:N", scale=None))
        .properties(width=400, height=400)
    )
    mo.hstack([pca_mat, pca_chart])
    return


# ---------------------------------------------------------------------------
# HoverZoom
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## HoverZoom

    A magnifying-glass overlay for any image or matplotlib figure.
    Hover over the chart to read labels on densely packed points.
    """)
    return


@app.cell
def _(np, plt):
    rng = np.random.default_rng(42)
    N_POINTS = 200
    hz_x = rng.normal(0, 1, N_POINTS)
    hz_y = rng.normal(0, 1, N_POINTS)
    labels = [f"p{i}" for i in range(N_POINTS)]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(hz_x, hz_y, s=12, alpha=0.6)
    for xi, yi, label in zip(hz_x, hz_y, labels):
        ax.annotate(label, (xi, yi), fontsize=4, alpha=0.7, ha="center", va="bottom")
    ax.set_title(f"{N_POINTS} labeled points — hover to read the labels")
    fig.tight_layout()
    return (fig,)


@app.cell
def _(HoverZoom, fig, mo):
    chart_widget = mo.ui.anywidget(HoverZoom(fig, zoom_factor=4.0, width=500))
    chart_widget
    return


# ---------------------------------------------------------------------------
# TextCompare
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TextCompare

    A side-by-side text comparison widget that highlights matching passages between two texts.
    Useful for plagiarism detection, finding shared passages, or comparing document versions.
    Hover over a highlighted match in one panel to see the corresponding match in the other.
    """)
    return


@app.cell
def _(TextCompare, mo):
    text_a = """The quick brown fox jumps over the lazy dog.
    This is a unique sentence in text A.
    Both texts share this common passage here.
    Another unique line for the first text."""

    text_b = """A quick brown fox leaps over a lazy dog.
    This is different content in text B.
    Both texts share this common passage here.
    Some other unique content for text B."""

    compare = mo.ui.anywidget(TextCompare(text_a=text_a, text_b=text_b, min_match_words=3))
    compare
    return compare, text_a, text_b


@app.cell
def _(compare, mo):
    mo.md(f"**Found {len(compare.matches)} matching passages**")
    return


if __name__ == "__main__":
    app.run()

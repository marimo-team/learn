import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Visualization

    In the previous episode, we selected photometry data from Pan-STARRS and used
    it to identify stars we think are likely to be in GD-1.

    In this episode, we will take the results from previous episodes and use them
    to make a figure that tells a compelling scientific story.

    ## Outline

    1. Starting with the figure from the previous episode, we will add annotations
       to present the results more clearly.

    2. Then we will learn several ways to customize figures to make them more
       appealing and effective.

    3. Finally, we will learn how to make a figure with multiple panels.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon

    from episode_functions import *
    return Polygon, np, pd, plt


@app.cell
def _(pd):
    filename = 'gd1_data.hdf'
    winner_df = pd.read_hdf(filename, 'winner_df')
    centerline_df = pd.read_hdf(filename, 'centerline_df')
    candidate_df = pd.read_hdf(filename, 'candidate_df')
    loop_df = pd.read_hdf(filename, 'loop_df')
    return candidate_df, centerline_df, filename, loop_df, winner_df


@app.cell
def _():
    pm1_min = -8.9
    pm1_max = -6.9
    pm2_min = -2.2
    pm2_max = 1.0

    pm1_rect, pm2_rect = make_rectangle(
        pm1_min, pm1_max, pm2_min, pm2_max)
    return pm1_max, pm1_min, pm1_rect, pm2_max, pm2_min, pm2_rect


@app.cell
def _(mo):
    mo.md(r"""
    ## Making Figures That Tell a Story

    The figures we have made so far have been "quick and dirty". Mostly we have used
    Matplotlib's default style, although we have adjusted a few parameters, like
    `markersize` and `alpha`, to improve legibility.

    Now that the analysis is done, it is time to think more about:

    1. Making professional-looking figures that are ready for publication.
    2. Making figures that communicate a scientific result clearly and compellingly.

    We will start by reviewing Figure 1 from the original paper, which has four panels
    and a caption.

    ## Exercise (10 minutes)

    Think about the following questions:

    1. What is the primary scientific result of this work?
    2. What story is this figure telling?
    3. In the design of this figure, can you identify 1 or 2 choices the authors made
       that you think are effective? Think about big-picture elements, like the number
       of panels and how they are arranged, as well as details like the choice of typeface.
    4. Can you identify 1 or 2 elements that could be improved, or that you might have
       done differently?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
No figure is perfect, and everyone can be a critic. Here are some topics that could come up in this discussion:

1. The primary result is that adding physical selection criteria makes it possible to separate likely candidates from the background more effectively than in previous work, which makes it possible to see the structure of GD-1 in "unprecedented detail," allowing the authors to detect that the stream is larger than previously observed.

2. The figure documents the selection process as a sequence of reproducible steps, containing enough information for a skeptical reader to understand the authors' choices. Reading right-to-left, top-to-bottom, we see selection based on proper motion, the results of the first selection, selection based on stellar surface properties (color and magnitude), and the results of the second selection.

3. The figure is mostly black and white, with minimal use of color, and mostly uses large fonts. It will likely work well in print. The annotations in the bottom left panel guide the reader to the results discussed in the text.

4. The panels that can have the same units, dimensions, and their axes are aligned, do. The on-sky positions likely do not need so much white space. Axes ticks for the on-sky position figures are not necessary since this is not in an intuitive coordinate system.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Plotting GD-1 with Annotations

    The lower left panel in the paper uses three other features to present the
    results more clearly and compellingly:

    - A vertical dashed line to distinguish the previously undetected region of GD-1,
    - A label that identifies the new region, and
    - Several annotations that combine text and arrows to identify features of GD-1.

    ## Exercise (20 minutes)

    Plot the selected stars in `winner_df` using the `plot_cmd_selection` function
    and then choose any or all of these features and add them to the figure:

    - To draw vertical lines, see `plt.vlines` and `plt.axvline`.
    - To add text, see `plt.text`.
    - To add an annotation with text and an arrow, see `plt.annotate`.
    """)
    return


@app.cell
def _(mo, plt, winner_df):
    mo.accordion({
        "Solution": mo.md(r"""
```python
fig = plt.figure(figsize=(10, 2.5))
plot_cmd_selection(winner_df)
plt.axvline(-55, ls='--', color='gray',
            alpha=0.4, dashes=(6, 4), lw=2)
plt.text(-60, 5.5, 'Previously\nundetected',
         fontsize='small', ha='right', va='top')

arrowprops = dict(color='gray', shrink=0.05, width=1.5,
                  headwidth=6, headlength=8, alpha=0.4)

plt.annotate('Spur', xy=(-33, 2), xytext=(-35, 5.5),
             arrowprops=arrowprops,
             fontsize='small')

plt.annotate('Gap', xy=(-22, -1), xytext=(-25, -5.5),
             arrowprops=arrowprops,
             fontsize='small')
```
        """)
    })
    return


@app.cell
def _(plt, winner_df):
    fig_annotated = plt.figure(figsize=(10, 2.5))
    plot_cmd_selection(winner_df)
    plt.axvline(-55, ls='--', color='gray',
                alpha=0.4, dashes=(6, 4), lw=2)
    plt.text(-60, 5.5, 'Previously\nundetected',
             fontsize='small', ha='right', va='top')

    arrowprops = dict(color='gray', shrink=0.05, width=1.5,
                      headwidth=6, headlength=8, alpha=0.4)

    plt.annotate('Spur', xy=(-33, 2), xytext=(-35, 5.5),
                 arrowprops=arrowprops,
                 fontsize='small')

    plt.annotate('Gap', xy=(-22, -1), xytext=(-25, -5.5),
                 arrowprops=arrowprops,
                 fontsize='small')
    return arrowprops, fig_annotated


@app.cell
def _(mo):
    mo.md(r"""
    ## Customization

    Matplotlib provides a default style that determines things like the colors of
    lines, the placement of labels and ticks on the axes, and many other properties.

    There are several ways to override these defaults and customize your figures:

    - To customize only the current figure, you can call functions like `tick_params`.
    - To customize all figures in a notebook, you can use `rcParams`.
    - To override more than a few defaults at the same time, you can use a style sheet.

    As a simple example, notice that Matplotlib puts ticks on the outside of the
    figures by default, and only on the left and bottom sides of the axes.

    **Note on Accessibility:** Customization offers a high degree of personalization
    for creating scientific visualizations. It is important to also create accessible
    visualizations for a broad audience that may include low-vision or color-blind
    individuals. The AAS Journals provide a Graphics Guide for authors with tips and
    external links.

    So far, everything we have wanted to do we could call directly from the pyplot
    module with `plt.`. As you do more and more customization you may need to run
    some methods on plotting objects themselves. To use the method that changes the
    direction of the ticks we need an `axes` object.

    To explicitly create an `axes` object we can first create our `figure` object
    and then add an `axes` object to it.
    """)
    return


@app.cell
def _(plt):
    fig_tick = plt.figure(figsize=(10, 2.5))
    ax_tick = fig_tick.add_subplot(1, 1, 1)
    ax_tick.tick_params(direction='in')
    return ax_tick, fig_tick


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    Read the documentation of `tick_params` and use it to put ticks on the top
    and right sides of the axes.
    """)
    return


@app.cell
def _(mo, plt):
    mo.accordion({
        "Solution": mo.md(r"""
```python
fig = plt.figure(figsize=(10, 2.5))
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(top=True, right=True)
```
        """)
    })
    return


@app.cell
def _(plt):
    fig_ticks_topright = plt.figure(figsize=(10, 2.5))
    ax_ticks_topright = fig_ticks_topright.add_subplot(1, 1, 1)
    ax_ticks_topright.tick_params(top=True, right=True)
    return ax_ticks_topright, fig_ticks_topright


@app.cell
def _(mo):
    mo.md(r"""
    ## rcParams

    If you want to make a customization that applies to all figures in a notebook,
    you can use `rcParams`. When you import Matplotlib, a dictionary is created with
    default values for everything you can change about your plot.

    Here is an example that reads the current font size from `rcParams`.
    """)
    return


@app.cell
def _(plt):
    plt.rcParams['font.size']
    return


@app.cell
def _(plt):
    plt.rcParams['font.size'] = 14
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    Plot the previous figure again, and see what font sizes have changed. Look up any
    other element of `rcParams`, change its value, and check the effect on the figure.
    """)
    return


@app.cell
def _(mo, plt):
    mo.accordion({
        "Solution": mo.md(r"""
```python
fig = plt.figure(figsize=(10, 2.5))
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(top=True, right=True)

# Looking up the 'axes.edgecolor' rcParams value
print(plt.rcParams['axes.edgecolor'])

plt.rcParams['axes.edgecolor'] = 'red'

fig = plt.figure(figsize=(10, 2.5))
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(top=True, right=True)

# Changing the rcParams value back to its original value
plt.rcParams['axes.edgecolor'] = 'black'
```
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    When you import Matplotlib, `plt.rcParams` is populated from a matplotlibrc file.
    If you want to permanently change a setting for every plot you make, you can set
    that in your matplotlibrc file. To find out where your matplotlibrc file lives:

    ```python
    import matplotlib as mpl
    mpl.matplotlib_fname()
    ```

    ## Style sheets

    It is possible that you would like multiple sets of defaults, for example, one
    default for plots for scientific papers and another for talks or posters.

    Matplotlib provides a set of predefined style sheets, or you can make your own.
    You can display a list of style sheets installed on your system.
    """)
    return


@app.cell
def _(plt):
    plt.style.available
    return


@app.cell
def _(mo):
    mo.md(r"""
    Note that `seaborn-paper`, `seaborn-talk` and `seaborn-poster` are particularly
    intended to prepare versions of a figure with text sizes and other features that
    work well in papers, talks, and posters.

    To use any of these style sheets, run `plt.style.use` like this.
    """)
    return


@app.cell
def _(plt):
    plt.style.use('fivethirtyeight')
    return


@app.cell
def _(mo):
    mo.md(r"""
    The style sheet you choose will affect the appearance of all figures you plot
    after calling `use`, unless you override any of the options or call `use` again.

    **Return to Default:** To switch back to the default style use:
    ```python
    plt.style.use('default')
    ```

    ## Exercise (5 minutes)

    Choose one of the styles on the list and select it by calling `use`. Then go
    back and plot one of the previous figures to see what changes in the figure's
    appearance.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
plt.style.use('seaborn-bright')

plot_cmd(candidate_df)
plt.plot(loop_df['color_loop'], loop_df['mag_loop'], label='GD1 Isochrone loop')

plt.legend()
```
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    If you cannot find a style sheet that is exactly what you want, you can make
    your own. This repository includes a style sheet called `az-paper-twocol.mplstyle`,
    with customizations chosen by Azalee Bostroem for publication in astronomy journals.

    You can use it like this:

    ```python
    plt.style.use('./az-paper-twocol.mplstyle')

    plot_cmd(candidate_df)
    plt.plot(loop_df['color_loop'], loop_df['mag_loop'], label='GD1 Isochrone loop')
    plt.legend()
    ```

    The prefix `./` tells Matplotlib to look for the file in the current directory.

    ## Multiple panels

    So far we have been working with one figure at a time, but the figure we are
    replicating contains multiple panels. We will create each of these panels as a
    different subplot.

    Matplotlib has multiple functions for making figures with multiple panels. We
    have already used `add_subplot` - however, this creates equal sized panels.
    For this reason, we will use `subplot2grid` which allows us to control the
    relative sizes of the panels.

    Like `add_subplot`, `subplot2grid` requires us to specify the total number of
    columns and rows in the grid (this time as a tuple called `shape`), and the
    location of the subplot (`loc`) - a tuple identifying the location in the grid
    we are about to fill.

    In this example, `shape` is `(2, 2)` to create two rows and two columns.
    For the first panel, `loc` is `(0, 0)`, which indicates row 0 and column 0,
    which is the upper-left panel.

    Here is how we use this function to draw the four panels.
    """)
    return


@app.cell
def _(candidate_df, centerline_df, plt, winner_df):
    plt.style.use('default')

    fig_equal = plt.figure()
    shape_equal = (2, 2)
    ax1_equal = plt.subplot2grid(shape_equal, (0, 0))
    plot_pm_selection(candidate_df)

    ax2_equal = plt.subplot2grid(shape_equal, (0, 1))
    plot_proper_motion(centerline_df)

    ax3_equal = plt.subplot2grid(shape_equal, (1, 0))
    plot_cmd_selection(winner_df)

    ax4_equal = plt.subplot2grid(shape_equal, (1, 1))
    plot_cmd(candidate_df)

    plt.tight_layout()
    return ax1_equal, ax2_equal, ax3_equal, ax4_equal, fig_equal, shape_equal


@app.cell
def _(mo):
    mo.md(r"""
    We use `plt.tight_layout` at the end, which adjusts the sizes of the panels to
    make sure the titles and axis labels don't overlap. Notice how convenient it is
    that we have written functions to plot each panel. This code is concise and
    readable.

    ## Exercise (5 minutes)

    What happens if you leave out `tight_layout`?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
Without `tight_layout` the space between the panels is too small. In this situation,
the titles from the lower plots overlap with the x-axis labels from the upper panels
and the axis labels from the right-hand panels overlap with the plots in the left-hand
panels.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Adjusting proportions

    In the previous figure, the panels are all the same size. To get a better view
    of GD-1, we would like to stretch the panels on the left and compress the ones
    on the right.

    To do that, we will use the `colspan` argument to make a panel that spans multiple
    columns in the grid. To do this we will need more columns so we will change the
    `shape` from (2,2) to (2,4).

    The panels on the left span three columns, so they are three times wider than the
    panels on the right.

    At the same time, we use `figsize` to adjust the aspect ratio of the whole figure.
    """)
    return


@app.cell
def _(candidate_df, centerline_df, plt, winner_df):
    plt.figure(figsize=(9, 4.5))

    shape_adj = (2, 4)
    ax1_adj = plt.subplot2grid(shape_adj, (0, 0), colspan=3)
    plot_pm_selection(candidate_df)

    ax2_adj = plt.subplot2grid(shape_adj, (0, 3))
    plot_proper_motion(centerline_df)

    ax3_adj = plt.subplot2grid(shape_adj, (1, 0), colspan=3)
    plot_cmd_selection(winner_df)

    ax4_adj = plt.subplot2grid(shape_adj, (1, 3))
    plot_cmd(candidate_df)

    plt.tight_layout()
    return ax1_adj, ax2_adj, ax3_adj, ax4_adj, shape_adj


@app.cell
def _(mo):
    mo.md(r"""
    This is looking more and more like the figure in the paper.

    ## Exercise (5 minutes)

    In this example, the ratio of the widths of the panels is 3:1. How would you
    adjust it if you wanted the ratio to be 3:2?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
plt.figure(figsize=(9, 4.5))

shape = (2, 5)                                    # CHANGED
ax1 = plt.subplot2grid(shape, (0, 0), colspan=3)
plot_pm_selection(candidate_df)

ax2 = plt.subplot2grid(shape, (0, 3), colspan=2)  # CHANGED
plot_proper_motion(centerline_df)

ax3 = plt.subplot2grid(shape, (1, 0), colspan=3)
plot_cmd_selection(winner_df)

ax4 = plt.subplot2grid(shape, (1, 3), colspan=2)  # CHANGED
plot_cmd(candidate_df)

plt.tight_layout()
```
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Adding the shaded regions

    The one thing our figure is missing is the shaded regions showing the stars
    selected by proper motion and around the isochrone in the color magnitude diagram.

    In episode 4 we defined a rectangle in proper motion space around the stars in
    GD-1. We stored the x-values of the vertices of this rectangle in `pm1_rect`
    and the y-values as `pm2_rect`.

    To plot this rectangle, we will use the Matplotlib `Polygon` object which we
    used in episode 7 to check which points were inside the polygon. However, this
    time we will be plotting the `Polygon`.

    To create a `Polygon`, we have to put the coordinates of the rectangle in an
    array with `x` values in the first column and `y` values in the second column.
    """)
    return


@app.cell
def _(np, pm1_rect, pm2_rect):
    vertices = np.transpose([pm1_rect, pm2_rect])
    vertices
    return (vertices,)


@app.cell
def _(mo):
    mo.md(r"""
    We will now create the `Polygon`, specifying its display properties which will be
    used when it is plotted. We will specify `closed=True` to make sure the shape is
    closed, `facecolor='orange'` to color the inside of the `Polygon` orange, and
    `alpha=0.4` to make the `Polygon` semi-transparent.

    Then to plot the `Polygon` we call the `add_patch` method. `add_patch` like
    `tick_params` must be called on an `axes` or `subplot` object.
    """)
    return


@app.cell
def _(Polygon, plt, vertices):
    fig_poly = plt.figure()
    ax_poly = fig_poly.add_subplot(1, 1, 1)
    poly_ex = Polygon(vertices, closed=True,
                      facecolor='orange', alpha=0.4)
    ax_poly.add_patch(poly_ex)
    ax_poly.set_xlim(-10, 7.5)
    ax_poly.set_ylim(-10, 10)
    return ax_poly, fig_poly, poly_ex


@app.cell
def _(mo):
    mo.md(r"""
    We can now call our `plot_proper_motion` function to plot the proper motion for
    each star, and then add a shaded `Polygon` to show the region we selected.
    """)
    return


@app.cell
def _(Polygon, centerline_df, plt, vertices):
    fig_pm = plt.figure()
    ax_pm = fig_pm.add_subplot(1, 1, 1)
    plot_proper_motion(centerline_df)
    poly_pm = Polygon(vertices, closed=True,
                      facecolor='C1', alpha=0.4)
    ax_pm.add_patch(poly_pm)
    return ax_pm, fig_pm, poly_pm


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    Add a few lines to be run after the `plot_cmd` function to show the polygon we
    selected as a shaded area.

    Hint: pass `loop_df` as an argument to `Polygon` as we did in episode 7 and then
    plot it using `add_patch`.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plot_cmd(candidate_df)
poly_cmd = Polygon(loop_df, closed=True,
              facecolor='C1', alpha=0.4)
ax.add_patch(poly_cmd)
```
        """)
    })
    return


@app.cell
def _(Polygon, candidate_df, loop_df, plt, plot_cmd):
    fig_cmd_poly = plt.figure()
    ax_cmd_poly = fig_cmd_poly.add_subplot(1, 1, 1)
    plot_cmd(candidate_df)
    poly_cmd = Polygon(loop_df, closed=True,
                       facecolor='C1', alpha=0.4)
    ax_cmd_poly.add_patch(poly_cmd)
    return ax_cmd_poly, fig_cmd_poly, poly_cmd


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise (5 minutes)

    Add the `Polygon` patches you just created to the right panels of the four
    panel figure.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Solution": mo.md(r"""
```python
fig = plt.figure(figsize=(9, 4.5))

shape = (2, 4)
ax1 = plt.subplot2grid(shape, (0, 0), colspan=3)
plot_pm_selection(candidate_df)

ax2 = plt.subplot2grid(shape, (0, 3))
plot_proper_motion(centerline_df)
poly = Polygon(vertices, closed=True,
               facecolor='orange', alpha=0.4)
ax2.add_patch(poly)

ax3 = plt.subplot2grid(shape, (1, 0), colspan=3)
plot_cmd_selection(winner_df)

ax4 = plt.subplot2grid(shape, (1, 3))
plot_cmd(candidate_df)
poly_cmd = Polygon(loop_df, closed=True,
               facecolor='orange', alpha=0.4)
ax4.add_patch(poly_cmd)

plt.tight_layout()
```
        """)
    })
    return


@app.cell
def _(Polygon, candidate_df, centerline_df, loop_df, plt, vertices, winner_df):
    fig_final = plt.figure(figsize=(9, 4.5))

    shape_final = (2, 4)
    ax1_final = plt.subplot2grid(shape_final, (0, 0), colspan=3)
    plot_pm_selection(candidate_df)

    ax2_final = plt.subplot2grid(shape_final, (0, 3))
    plot_proper_motion(centerline_df)
    poly_final = Polygon(vertices, closed=True,
                         facecolor='orange', alpha=0.4)
    ax2_final.add_patch(poly_final)

    ax3_final = plt.subplot2grid(shape_final, (1, 0), colspan=3)
    plot_cmd_selection(winner_df)

    ax4_final = plt.subplot2grid(shape_final, (1, 3))
    plot_cmd(candidate_df)
    poly_cmd_final = Polygon(loop_df, closed=True,
                             facecolor='orange', alpha=0.4)
    ax4_final.add_patch(poly_cmd_final)

    plt.tight_layout()
    return (
        ax1_final,
        ax2_final,
        ax3_final,
        ax4_final,
        fig_final,
        poly_cmd_final,
        poly_final,
        shape_final,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    In this episode, we reverse-engineered the figure we have been replicating,
    identifying elements that seem effective and others that could be improved.

    We explored features Matplotlib provides for adding annotations to figures --
    including text, lines, arrows, and polygons -- and several ways to customize
    the appearance of figures. And we learned how to create figures that contain
    multiple panels.

    **Key points:**

    - Effective figures focus on telling a single story clearly and authentically.
      The major decisions needed in creating an effective summary figure like this
      one can be done away from a computer and built up from low fidelity (hand drawn)
      to high (tweaking rcParams, etc.).
    - Consider using annotations to guide the reader's attention to the most important
      elements of a figure, while keeping in mind accessibility issues that such detail
      may introduce.
    - The default Matplotlib style generates good quality figures, but there are several
      ways you can override the defaults.
    - If you find yourself making the same customizations on several projects, you might
      want to create your own style sheet.
    """)
    return

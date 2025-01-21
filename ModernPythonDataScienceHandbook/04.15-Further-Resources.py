import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Further Resources

        A single part of a book can never hope to cover all the available features and plot types available in Matplotlib.
        As with other packages we've seen, liberal use of IPython's tab completion and help functions (see [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)) can be very helpful when exploring Matplotlib's API.
        In addition, Matplotlib’s [online documentation](http://matplotlib.org/) can be a helpful reference.
        See in particular the [Matplotlib gallery](https://matplotlib.org/stable/gallery/), which shows thumbnails of hundreds of different plot types, each one linked to a page with the Python code snippet used to generate it.
        This allows you to visually inspect and learn about a wide range of different plotting styles and visualization techniques.

        For a book-length treatment of Matplotlib, I would recommend *Interactive Applications Using Matplotlib* (Packt), written by Matplotlib core developer Ben Root.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Other Python Visualization Libraries

        Although Matplotlib is the most prominent Python visualization library, there are other more modern tools that are worth exploring as well.
        I'll mention a few of them briefly here:

        - [Bokeh](http://bokeh.pydata.org) is a JavaScript visualization library with a Python frontend that creates highly interactive visualizations capable of handling very large and/or streaming datasets.
        - [Plotly](http://plot.ly) is the eponymous open source product of the Plotly company, and is similar in spirit to Bokeh. It is actively developed and provides a wide range of interactive chart types.
        - [HoloViews](https://holoviews.org/) is a more declarative, unified API for generating charts in a variety of backends, including Bokeh and Matplotlib.
        - [Vega](https://vega.github.io/) and [Vega-Lite](https://vega.github.io/vega-lite) are declarative graphics representations, and are the product of years of research into how to think about data visualization and interaction. The reference rendering implementation is JavaScript, and the [Altair package](https://altair-viz.github.io/) provides a Python API to generate these charts.

        The visualization landscape in the Python world is constantly evolving, and I expect that this list may be out of date by the time this book is published.
        Additionally, because Python is used in so many domains, you'll find many other visualization tools built for more specific use cases.
        It can be hard to keep track of all of them, but a good resource for learning about this wide variety of visualization tools is https://pyviz.org/, an open, community-driven site containing tutorials and examples of many different visualization tools.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

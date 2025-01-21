import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # More IPython Resources
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this set of chapters, we've just scratched the surface of using IPython to enable data science tasks.
        Much more information is available both in print and on the web, and here I'll list some other resources that you may find helpful.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Web Resources

        - [The IPython website](http://ipython.org): The IPython website provides links to documentation, examples, tutorials, and a variety of other resources.
        - [The nbviewer website](http://nbviewer.jupyter.org/): This site shows static renderings of any Jupyter notebook available on the internet. The front page features some example notebooks that you can browse to see what other folks are using IPython for!
        - [A curated collection of Jupyter notebooks](https://github.com/jupyter/jupyter/wiki): This ever-growing list of notebooks, powered by nbviewer, shows the depth and breadth of numerical analysis you can do with IPython. It includes everything from short examples and tutorials to full-blown courses and books composed in the notebook format!
        - Video tutorials: Searching the internet, you will find many video tutorials on IPython. I'd especially recommend seeking tutorials from the PyCon, SciPy, and PyData conferences by Fernando Perez and Brian Granger, two of the primary creators and maintainers of IPython and Jupyter.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Books

        - [*Python for Data Analysis* (O'Reilly)](http://shop.oreilly.com/product/0636920023784.do): Wes McKinney's book includes a chapter that covers using IPython as a data scientist. Although much of the material overlaps what we've discussed here, another perspective is always helpful.
        - [*Learning IPython for Interactive Computing and Data Visualization* (Packt)](https://www.packtpub.com/big-data-and-business-intelligence/learning-ipython-interactive-computing-and-data-visualization): This short book by Cyrille Rossant offers a good introduction to using IPython for data analysis.
        - [*IPython Interactive Computing and Visualization Cookbook* (Packt)](https://www.packtpub.com/big-data-and-business-intelligence/ipython-interactive-computing-and-visualization-cookbook): Also by Cyrille Rossant, this book is a longer and more advanced treatment of using IPython for data science. Despite its name, it's not just about IPython; it also goes into some depth on a broad range of data science topics.

        Finally, a reminder that you can find help on your own: IPython's `?`-based help functionality (discussed in [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)) can be useful if you use it well and use it often.
        As you go through the examples here and elsewhere, this can be used to familiarize yourself with all the tools that IPython has to offer.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

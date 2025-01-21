import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Machine Learning
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This final part is an introduction to the very broad topic of machine learning, mainly via Python's [Scikit-Learn](http://scikit-learn.org) package.
        You can think of machine learning as a class of algorithms that allow a program to detect particular patterns in a dataset, and thus "learn" from the data to draw inferences from it.
        This is not meant to be a comprehensive introduction to the field of machine learning; that is a large subject and necessitates a more technical approach than we take here.
        Nor is it meant to be a comprehensive manual for the use of the Scikit-Learn package (for this, you can refer to the resources listed in [Further Machine Learning Resources](05.15-Learning-More.ipynb)).
        Rather, the goals here are:

        - To introduce the fundamental vocabulary and concepts of machine learning
        - To introduce the Scikit-Learn API and show some examples of its use
        - To take a deeper dive into the details of several of the more important classical machine learning approaches, and develop an intuition into how they work and when and where they are applicable

        Much of this material is drawn from the Scikit-Learn tutorials and workshops I have given on several occasions at PyCon, SciPy, PyData, and other conferences.
        Any clarity in the following pages is likely due to the many workshop participants and co-instructors who have given me valuable feedback on this material over the years!
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

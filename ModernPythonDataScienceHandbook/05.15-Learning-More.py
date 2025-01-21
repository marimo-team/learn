import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Further Machine Learning Resources
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This part of the book has been a quick tour of machine learning in Python, primarily using the tools within the Scikit-Learn library.
        As long as these chapters are, they are still too short to cover many interesting and important algorithms, approaches, and discussions.
        Here I want to suggest some resources to learn more about machine learning in Python, for those who are interested:

        - [The Scikit-Learn website](http://scikit-learn.org): The Scikit-Learn website has an impressive breadth of documentation and examples covering some of the models discussed here, and much, much more. If you want a brief survey of the most important and often-used machine learning algorithms, this is a good place to start.

        - *SciPy, PyCon, and PyData tutorial videos*: Scikit-Learn and other machine learning topics are perennial favorites in the tutorial tracks of many Python-focused conference series, in particular the PyCon, SciPy, and PyData conferences. Most of these conferences publish videos of their keynotes, talks, and tutorials for free online, and you should be able to find these easily via a suitable web search (for example, "PyCon 2022 videos").

        - [*Introduction to Machine Learning with Python*](http://shop.oreilly.com/product/0636920030515.do), by Andreas C. Müller and Sarah Guido (O'Reilly). This book covers many of the machine learning fundamentals discussed in these chapters, but is particularly relevant for its coverage of more advanced features of Scikit-Learn, including additional estimators, model validation approaches, and pipelining.

        - [*Machine Learning with PyTorch and Scikit-Learn*](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312), by Sebastian Raschka (Packt). Sebastian Raschka's most recent book starts with some of the fundamental topics covered in these chapters, but goes deeper and shows how those concepts apply to more sophisticated and computationally intensive deep learning and reinforcement learning models using the well-known [PyTorch library](https://pytorch.org/).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

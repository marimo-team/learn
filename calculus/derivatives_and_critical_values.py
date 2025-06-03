import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Derivatives and Critical Values

    _By [ghimiresunil](https://github.com/ghimiresunil)

    Welcome to this tutorial on **Derivatives and Critical Values**—a key turning point in our exploration of calculus. In the previous module, we learned how to describe the *rate of change* of a function. In this notebook, we take that understanding further to analyze how functions behave, where they reach their highest or lowest values, and how we can use calculus to find those points.

    ---

    ## Quick Recap: Rates of change

    In the **Rates of Change** notebook, we explored two foundational ideas:

    - **Average Rate of Change**:  
      This is the change in a function’s output relative to the change in input over an interval. It gives us a *global* view of how a function behaves between two points.

    \[ \text{Average Rate of Change} = \frac{f(b) - f(a)}{b - a} \]

    - **Instantaneous Rate of Change**:  
      As the interval becomes infinitesimally small, the average rate of change approaches the **derivative**—a *local* measure of how the function is changing at a specific point.

   
    \[f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h} \]

    This transition from global to local change is where derivatives begin to reveal the fine structure of functions, helping us uncover turning points, slopes, and trends.

    ---

    ## What You'll Learn in This Notebook

    In this notebook, we will focus on:

    - What *critical values* are and how to find them  
    - How to use the **first derivative** to identify increasing/decreasing behavior  
    - How to apply the **second derivative** to determine concavity and locate maxima or minima  
    - How these tools help us analyze and sketch functions  
    - How this foundation prepares us for solving real-world **optimization problems**

    By the end of this module, you’ll be able to use derivatives not just to describe change, but to *predict, optimize, and explain* key behaviors in a variety of systems—from economics to physics to machine learning.

    Let’s get started...
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import Necessary Libraries""")
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sympy import diff, symbols
    return (mo,)


if __name__ == "__main__":
    app.run()

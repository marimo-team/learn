# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cvxpy==1.6.0",
#     "marimo",
#     "numpy==2.2.2",
# ]
# ///

import marimo

__generated_with = "0.11.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Convex optimization

        In the previous tutorials, we learned about least squares, linear programming,
        and quadratic programming, and saw applications of each. We also learned that these problem
        classes can be solved efficiently and reliably using CVXPY. That's because these problem classes are a special
        case of a more general class of tractable problems, called **convex optimization problems.**

        A convex optimization problem is an optimization problem that minimizes a convex
        function, subject to affine equality constraints and convex inequality
        constraints ($f_i(x)\leq 0$, where $f_i$ is a convex function).

        **CVXPY.** CVXPY lets you specify and solve any convex optimization problem,
        abstracting away the more specific problem classes. You start with CVXPY's **atomic functions**, like `cp.exp`, `cp.log`, and `cp.square`, and compose them to build more complex convex functions. As long as the functions are composed in the right way — as long as they are "DCP-compliant" —  your resulting problem will be convex and solvable by CVXPY.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **🛑 Stop!** Before proceeding, read the CVXPY docs to learn about atomic functions and the DCP ruleset:

        https://www.cvxpy.org/tutorial/index.html
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Is my problem DCP-compliant?** Below is a sample CVXPY problem. It is DCP-compliant. Try typing in other problems and seeing if they are DCP-compliant. If you know your problem is convex, there exists a way to express it in a DCP-compliant way.""")
    return


@app.cell
def _(mo):
    import cvxpy as cp
    import numpy as np

    x = cp.Variable(3)
    P_sqrt = np.random.randn(3, 3)

    objective = cp.log(np.random.randn(3) @ x) - cp.sum_squares(P_sqrt @ x)
    constraints = [x >= 0, cp.sum(x) == 1]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    mo.md(f"Is my problem DCP? `{problem.is_dcp()}`")
    return P_sqrt, constraints, cp, np, objective, problem, x


@app.cell
def _(problem):
    problem.solve()
    return


@app.cell
def _(x):
    x.value
    return


if __name__ == "__main__":
    app.run()

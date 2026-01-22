# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cvxpy==1.6.0",
#     "marimo",
#     "numpy==2.2.2",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Least squares

    In a least-squares problem, we have measurements $A \in \mathcal{R}^{m \times
    n}$ (i.e., $m$ rows and $n$ columns) and $b \in \mathcal{R}^m$. We seek a vector
    $x \in \mathcal{R}^{n}$ such that $Ax$ is close to $b$. The matrices $A$ and $b$ are problem data or constants, and $x$ is the variable we are solving for.

    Closeness is defined as the sum of the squared differences:

    \[ \sum_{i=1}^m (a_i^Tx - b_i)^2, \]

    also known as the $\ell_2$-norm squared, $\|Ax - b\|_2^2$.

    For example, we might have a dataset of $m$ users, each represented by $n$ features. Each row $a_i^T$ of $A$ is the feature vector for user $i$, while the corresponding entry $b_i$ of $b$ is the measurement we want to predict from $a_i^T$, such as ad spending. The prediction for user $i$ is given by $a_i^Tx$.

    We find the optimal value of $x$ by solving the optimization problem

    \[
        \begin{array}{ll}
        \text{minimize}   & \|Ax - b\|_2^2.
        \end{array}
    \]

    Let $x^\star$ denote the optimal $x$. The quantity $r = Ax^\star - b$ is known as the residual. If $\|r\|_2 = 0$, we have a perfect fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example

    In this example, we use the Python library [CVXPY](https://github.com/cvxpy/cvxpy) to construct and solve a least-squares problems.
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    return cp, np


@app.cell
def _():
    m = 20
    n = 15
    return m, n


@app.cell
def _(m, n, np):
    np.random.seed(0)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    return A, b


@app.cell
def _(A, b, cp, n):
    x = cp.Variable(n)
    objective = cp.sum_squares(A @ x - b)
    problem = cp.Problem(cp.Minimize(objective))
    optimal_value = problem.solve()
    return optimal_value, x


@app.cell
def _(A, b, cp, mo, optimal_value, x):
    mo.md(
        f"""
        - The optimal value is **{optimal_value:.04f}**.
        - The optimal value of $x$ is {mo.as_html(list(x.value))}
        - The norm of the residual is **{cp.norm(A @ x - b, p=2).value:0.4f}**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Further reading

    For a primer on least squares, with many real-world examples, check out the free book
    [Vectors, Matrices, and Least Squares](https://web.stanford.edu/~boyd/vmls/), which is used for undergraduate linear algebra education at Stanford.
    """)
    return


if __name__ == "__main__":
    app.run()

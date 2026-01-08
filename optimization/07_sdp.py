# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cvxpy==1.6.0",
#     "marimo",
#     "numpy==2.2.2",
#     "wigglystuff==0.1.9",
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
    # Semidefinite program
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _This notebook introduces an advanced topic._ A semidefinite program (SDP) is an optimization problem of the form

    \[
        \begin{array}{ll}
        \text{minimize}   & \mathbf{tr}(CX) \\
        \text{subject to} & \mathbf{tr}(A_iX) = b_i, \quad i=1,\ldots,p \\
                          & X \succeq 0,
        \end{array}
    \]

    where $\mathbf{tr}$ is the trace function, $X \in \mathcal{S}^{n}$ is the optimization variable and $C, A_1, \ldots, A_p \in \mathcal{S}^{n}$, and $b_1, \ldots, b_p \in \mathcal{R}$ are problem data, and $X \succeq 0$ is a matrix inequality. Here $\mathcal{S}^{n}$ denotes the set of $n$-by-$n$ symmetric matrices.

    **Example.** An example of an SDP is to complete a covariance matrix $\tilde \Sigma \in \mathcal{S}^{n}_+$ with missing entries $M \subset \{1,\ldots,n\} \times \{1,\ldots,n\}$:

    \[
        \begin{array}{ll}
        \text{minimize}   & 0 \\
        \text{subject to} & \Sigma_{ij} = \tilde \Sigma_{ij}, \quad (i,j) \notin M \\
                          & \Sigma \succeq 0,
        \end{array}
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example

    In the following code, we show how to specify and solve an SDP with CVXPY.
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    return cp, np


@app.cell
def _(np):
    # Generate a random SDP.
    n = 3
    p = 3
    np.random.seed(1)
    C = np.random.randn(n, n)
    A = []
    b = []
    for i in range(p):
        A.append(np.random.randn(n, n))
        b.append(np.random.randn())
    return A, C, b, n, p


@app.cell
def _(A, C, b, cp, n, p):
    # Create a symmetric matrix variable.
    X = cp.Variable((n, n), symmetric=True)

    # The operator >> denotes matrix inequality, with X >> 0 constraining X
    # to be positive semidefinite
    constraints = [X >> 0]
    constraints += [cp.trace(A[i] @ X) == b[i] for i in range(p)]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    _ = prob.solve()
    return X, prob


@app.cell
def _(X, mo, prob, wigglystuff):
    mo.md(
        f"""
        The optimal value is {prob.value:0.4f}.

        A solution for $X$ is (rounded to the nearest decimal) is: 

        {mo.ui.anywidget(wigglystuff.Matrix(X.value)).center()}
        """
    )
    return


@app.cell
def _():
    import wigglystuff
    return (wigglystuff,)


if __name__ == "__main__":
    app.run()

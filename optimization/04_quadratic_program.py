# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cvxpy==1.6.0",
#     "marimo",
#     "numpy==2.2.2",
# ]
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Quadratic program

        A quadratic program is an optimization problem with a quadratic objective and
        affine equality and inequality constraints. A common standard form is the
        following:

        \[
            \begin{array}{ll}
            \text{minimize}   & (1/2)x^TPx + q^Tx\\
            \text{subject to} & Gx \leq h \\
                              & Ax = b.
            \end{array}
        \]

        Here $P \in \mathcal{S}^{n}_+$, $q \in \mathcal{R}^n$, $G \in \mathcal{R}^{m \times n}$, $h \in \mathcal{R}^m$, $A \in \mathcal{R}^{p \times n}$, and $b \in \mathcal{R}^p$ are problem data and $x \in \mathcal{R}^{n}$ is the optimization variable. The inequality constraint $Gx \leq h$ is elementwise.

        **Why quadratic programming?** Quadratic programs are convex optimization problems that generalize both least-squares and linear programming.They can be solved efficiently and reliably, even in real-time.

        **An example from finance.** A simple example of a quadratic program arises in finance. Suppose we have $n$ different stocks, an estimate $r \in \mathcal{R}^n$ of the expected return on each stock, and an estimate $\Sigma \in \mathcal{S}^{n}_+$ of the covariance of the returns. Then we solve the optimization problem

        \[
            \begin{array}{ll}
            \text{minimize}   & (1/2)x^T\Sigma x - r^Tx\\
            \text{subject to} & x \geq 0 \\
                              & \mathbf{1}^Tx = 1,
            \end{array}
        \]

        to find a nonnegative portfolio allocation $x \in \mathcal{R}^n_+$ that optimally balances expected return and variance of return.

        When we solve a quadratic program, in addition to a solution $x^\star$, we obtain a dual solution $\lambda^\star$ corresponding to the inequality constraints. A positive entry $\lambda^\star_i$ indicates that the constraint $g_i^Tx \leq h_i$ holds with equality for $x^\star$ and suggests that changing $h_i$ would change the optimal value.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example

        In this example, we use CVXPY to construct and solve a quadratic program.
        """
    )
    return


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    return cp, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""First we generate synthetic data.""")
    return


@app.cell
def _(np):
    m = 15
    n = 10
    p = 5

    np.random.seed(1)
    P = np.random.randn(n, n)
    P = P.T @ P
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)
    return A, G, P, b, h, m, n, p, q


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we specify the problem. Notice that we use the `quad_form` function from CVXPY to create the quadratic form $x^TPx$.""")
    return


@app.cell
def _(A, G, P, b, cp, h, n, q):
    x = cp.Variable(n)

    problem = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x),
        [G @ x <= h, A @ x == b],
    )
    _ = problem.solve()
    return problem, x


@app.cell(hide_code=True)
def _(mo, problem, x):
    mo.md(
        f"""
        The optimal value is {problem.value:.04f}.

        A solution $x$ is {mo.as_html(list(x.value))}
        A dual solution is is {mo.as_html(list(problem.constraints[0].dual_value))}
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

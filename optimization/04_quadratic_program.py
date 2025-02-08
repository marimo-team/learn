# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cvxpy==1.6.0",
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.2",
#     "wigglystuff==0.1.9",
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
    mo.md("""First we generate synthetic data. In this problem, we don't include equality constraints, only inequality.""")
    return


@app.cell
def _(np):
    m = 4
    n = 2

    np.random.seed(1)
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    return G, h, m, n, q


@app.cell(hide_code=True)
def _(mo, np):
    import wigglystuff

    P_widget = mo.ui.anywidget(
        wigglystuff.Matrix(np.array([[4.0, -1.4], [-1.4, 4]]), step=0.1)
    )

    mo.md(
        f"""
        The quadratic form $P$ is equal to the symmetrized version of this
        matrix:

        {P_widget.center()}
        """
    )
    return P_widget, wigglystuff


@app.cell
def _(P_widget, np):
    P = 0.5 * (np.array(P_widget.matrix) + np.array(P_widget.matrix).T)
    return (P,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we specify the problem. Notice that we use the `quad_form` function from CVXPY to create the quadratic form $x^TPx$.""")
    return


@app.cell
def _(G, P, cp, h, n, q):
    x = cp.Variable(n)

    problem = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x),
        [G @ x <= h],
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
def _(G, P, h, plot_contours, q, x):
    plot_contours(P, G, h, q, x.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this plot, the gray shaded region is the feasible region (points satisfying the inequality), and the ellipses are level curves of the quadratic form.

        **🌊 Try it!** Try changing the entries of $P$ above with your mouse. How do the
        level curves and the optimal value of $x$ change? Can you explain what you see?
        """
    )
    return


@app.cell(hide_code=True)
def _(P, mo):
    mo.md(
        rf"""
        The above contour lines were generated with
        
        \[
        P= \begin{{bmatrix}}
        {P[0, 0]:.01f} & {P[0, 1]:.01f} \\
        {P[1, 0]:.01f} & {P[1, 1]:.01f} \\
        \end{{bmatrix}}
        \]
        """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def plot_contours(P, G, h, q, x_star):
        import matplotlib.pyplot as plt

        # Create a grid of x and y values.
        x = np.linspace(-5, 5, 400)
        y = np.linspace(-5, 5, 400)
        X, Y = np.meshgrid(x, y)

        # Compute the quadratic form Q(x, y) = a*x^2 + 2*b*x*y + c*y^2.
        # Here, a = P[0,0], b = P[0,1] (and P[1,0]), c = P[1,1]
        Z = (
            0.5 * (P[0, 0] * X**2 + 2 * P[0, 1] * X * Y + P[1, 1] * Y**2)
            + q[0] * X
            + q[1] * Y
        )

        # --- Evaluate the constraints on the grid ---
        # We stack X and Y to get a list of (x,y) points.
        points = np.vstack([X.ravel(), Y.ravel()]).T

        # Start with all points feasible
        feasible = np.ones(points.shape[0], dtype=bool)

        # Apply the inequality constraints Gx <= h.
        # Each row of G and corresponding h defines a condition.
        for i in range(G.shape[0]):
            # For a given point x, the condition is: G[i,0]*x + G[i,1]*y <= h[i]
            feasible &= points.dot(G[i]) <= h[i] + 1e-8  # small tolerance
        # Reshape the boolean mask back to grid shape.
        feasible_grid = feasible.reshape(X.shape)

        # --- Plot the feasible region and contour lines---
        plt.figure(figsize=(8, 6))

        # Use contourf to fill the region where feasible_grid is True.
        # We define two levels, so that points that are True (feasible) get one
        # color.
        plt.contourf(
            X,
            Y,
            feasible_grid,
            levels=[-0.5, 0.5, 1.5],
            colors=["white", "gray"],
            alpha=0.5,
        )

        contours = plt.contour(X, Y, Z, levels=10, cmap="viridis")
        plt.clabel(contours, inline=True, fontsize=8)
        plt.title("Feasible region and level curves")
        plt.xlabel("$x_1$")
        plt.ylabel("$y_2$")
        # plt.colorbar(contours, label='Q(x, y)')

        ax = plt.gca()
        # Optionally, mark and label the point x_star.
        ax.plot(x_star[0], x_star[1], "ko", markersize=5)
        ax.text(
            x_star[0],
            x_star[1],
            r"$\mathbf{x}^\star$",
            color="black",
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
        )
        return plt.gca()
    return (plot_contours,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

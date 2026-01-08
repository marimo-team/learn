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

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear program

    A linear program is an optimization problem with a linear objective and affine
    inequality constraints. A common standard form is the following:

    \[
        \begin{array}{ll}
        \text{minimize}   & c^Tx \\
        \text{subject to} & Ax \leq b.
        \end{array}
    \]

    Here $A \in \mathcal{R}^{m \times n}$, $b \in \mathcal{R}^m$, and $c \in \mathcal{R}^n$ are problem data and $x \in \mathcal{R}^{n}$ is the optimization variable. The inequality constraint $Ax \leq b$ is elementwise.

    For example, we might have $n$ different products, each constructed out of $m$ components. Each entry $A_{ij}$ is the amount of component $i$ required to build one unit of product $j$. Each entry $b_i$ is the total amount of component $i$ available. We lose $c_j$ for each unit of product $j$ ($c_j < 0$ indicates profit). Our goal then is to choose how many units of each product $j$ to make, $x_j$, in order to minimize loss without exceeding our budget for any component.

    In addition to a solution $x^\star$, we obtain a dual solution $\lambda^\star$. A positive entry $\lambda^\star_i$ indicates that the constraint $a_i^Tx \leq b_i$ holds with equality for $x^\star$ and suggests that changing $b_i$ would change the optimal value.

    **Why linear programming?** Linear programming is a way to achieve an optimal outcome, such as maximum utility or lowest cost, subject to a linear objective function and affine constraints. Developed in the 20th century, linear programming is widely used today to solve problems in resource allocation, scheduling, transportation, and more. The discovery of polynomial-time algorithms to solve linear programs was of tremendous worldwide importance and entered the public discourse, even making the front page of the New York Times.

    In the late 20th and early 21st century, researchers generalized linear programming to a much wider class of problems called convex optimization problems. Nearly all convex optimization problems can be solved efficiently and reliably, and even more difficult problems are readily solved by a sequence of convex optimization problems. Today, convex optimization is used to fit machine learning models, land rockets in real-time at SpaceX, plan trajectories for self-driving cars at Waymo, execute many billions of dollars of financial trades a day, and much more.

    This marimo learn course uses CVXPY, a modeling language for convex optimization problems developed originally at Stanford, to construct and solve convex programs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
        {mo.image("https://www.debugmind.com/wp-content/uploads/2020/01/spacex-1.jpg")}
        _SpaceX solves convex optimization problems onboard to land its rockets, using CVXGEN, a code generator for quadratic programming developed at Stephen Boyd’s Stanford lab. Photo by SpaceX, licensed CC BY-NC 2.0._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example

    Here we use CVXPY to construct and solve a linear program.
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    return cp, np


@app.cell(hide_code=True)
def _(np):
    A = np.array(
        [
            [0.76103773, 0.12167502],
            [0.44386323, 0.33367433],
            [1.49407907, -0.20515826],
            [0.3130677, -0.85409574],
            [-2.55298982, 0.6536186],
            [0.8644362, -0.74216502],
            [2.26975462, -1.45436567],
            [0.04575852, -0.18718385],
            [1.53277921, 1.46935877],
            [0.15494743, 0.37816252],
        ]
    )

    b = np.array(
        [
            2.05062369,
            0.94934659,
            0.89559424,
            1.04389978,
            2.45035643,
            -0.95479445,
            -0.83801349,
            -0.26562529,
            2.35763652,
            0.98286942,
        ]
    )
    return A, b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We've randomly generated problem data $A$ and $B$. The vector for $c$ is shown below. Try playing with the value of $c$ by dragging the components, and see how the level curves change in the visualization below.
    """)
    return


@app.cell
def _(mo, np):
    from wigglystuff import Matrix

    c_widget = mo.ui.anywidget(Matrix(matrix=np.array([[0.1, -0.2]]), step=0.01))
    c_widget
    return (c_widget,)


@app.cell
def _(c_widget, np):
    c = np.array(c_widget.value["matrix"][0])
    return (c,)


@app.cell
def _(A, b, c, cp):
    x = cp.Variable(A.shape[1])
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    _ = prob.solve()
    x_star = x.value
    return prob, x, x_star


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below, we plot the feasible region of the problem — the intersection of the inequalities — and the level curves of the objective function. The optimal value $x^\star$ is the point farthest in the feasible region in the direction $-c$.
    """)
    return


@app.cell(hide_code=True)
def _(A, b, c, make_plot, x_star):
    make_plot(A, b, c, x_star)
    return


@app.cell(hide_code=True)
def _(np):
    import matplotlib.pyplot as plt


    def make_plot(A, b, c, x_star):
        # Define a grid over a region that covers the feasible set.
        # You might need to adjust these limits.
        x_vals = np.linspace(-1, 1, 400)
        y_vals = np.linspace(1, 3, 400)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Flatten the grid points into an (N,2) array.
        points = np.vstack([X.ravel(), Y.ravel()]).T

        # For each point, check if it satisfies all the constraints: A @ x <= b.
        # A dot product: shape of A @ x.T will be (m, N). We add a little tolerance.
        feasible = np.all(np.dot(A, points.T) <= (b[:, None] + 1e-8), axis=0)
        feasible = feasible.reshape(X.shape)

        # Create the figure with a white background.
        fig = plt.figure(figsize=(8, 6), facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")

        # Plot the feasible region.
        # Since "feasible" is a boolean array (False=0, True=1), we set contour levels so that
        # the region with value 1 (feasible) gets filled.
        ax.contourf(
            X,
            Y,
            feasible.astype(float),
            levels=[-0.5, 0.5, 1.5],
            colors=["white", "gray"],
            alpha=0.5,
        )

        # Plot level curves of the objective function c^T x.
        # Compute c^T x over the grid:
        Z = c[0] * X + c[1] * Y
        # Choose several levels for the iso-cost lines.
        levels = np.linspace(np.min(Z), np.max(Z), 20)
        contours = ax.contour(
            X, Y, Z, levels=levels, colors="gray", linestyles="--", linewidths=1
        )

        # Draw the vector -c as an arrow starting at x_star.
        norm_c = np.linalg.norm(c)
        if norm_c > 0:
            head_width = norm_c * 0.1
            head_length = norm_c * 0.1
            # The arrow starts at x_star and points in the -c direction.
            ax.arrow(
                x_star[0],
                x_star[1],
                -c[0],
                -c[1],
                head_width=head_width,
                head_length=head_length,
                fc="black",
                ec="black",
                length_includes_head=True,
            )
            # Label the arrow near its tip.
            ax.text(
                x_star[0] - c[0] * 1.05,
                x_star[1] - c[1] * 1.05,
                r"$-c$",
                color="black",
                fontsize=12,
            )

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
        # Label the axes and set title.
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("Feasible Region and Level Curves of $c^Tx$")
        ax.set_xlim(np.min(x_vals), np.max(x_vals))
        ax.set_ylim(np.min(y_vals), np.max(y_vals))
        return ax
    return (make_plot,)


@app.cell(hide_code=True)
def _(mo, prob, x):
    mo.md(
        f"""
        The optimal value is {prob.value:.04f}.

        A solution $x$ is {mo.as_html(list(x.value))}
        A dual solution is is {mo.as_html(list(prob.constraints[0].dual_value))}
        """
    )
    return


if __name__ == "__main__":
    app.run()

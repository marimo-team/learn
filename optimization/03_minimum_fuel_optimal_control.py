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
    # Minimal fuel optimal control

    This notebook includes an application of linear programming to controlling a
    physical system, adapted from [Convex
    Optimization](https://web.stanford.edu/~boyd/cvxbook/) by Boyd and Vandenberghe.

    We consider a linear dynamical system with state $x(t) \in \mathbf{R}^n$, for $t = 0, \ldots, T$. At each time step $t = 0, \ldots, T - 1$, an actuator or input signal $u(t)$ is applied, affecting the state. The dynamics
    of the system is given by the linear recurrence

    \[
        x(t + 1) = Ax(t) + bu(t), \quad t = 0, \ldots, T - 1,
    \]

    where $A \in \mathbf{R}^{n \times n}$ and $b \in \mathbf{R}^n$ are given and encode how the system evolves. The initial state $x(0)$ is also given.

    The _minimum fuel optimal control problem_ is to choose the inputs $u(0), \ldots, u(T - 1)$ so as to achieve
    a given desired state $x_\text{des} = x(T)$ while minimizing the total fuel consumed

    \[
    F = \sum_{t=0}^{T - 1} f(u(t)).
    \]

    The function $f : \mathbf{R} \to \mathbf{R}$ tells us how much fuel is consumed as a function of the input, and is given by

    \[
        f(a) = \begin{cases}
        |a| & |a| \leq 1 \\
        2|a| - 1 & |a| > 1.
        \end{cases}
    \]

    This means the fuel use is proportional to the magnitude of the signal between $-1$ and $1$, but for larger signals the marginal fuel efficiency is half.

    **This notebook.** In this notebook we use CVXPY to formulate the minimum fuel optimal control problem as a linear program. The notebook lets you play with the initial and target states, letting you see how they affect the planned trajectory of inputs $u$.

    First, we create the **problem data**.
    """)
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _():
    n, T = 3, 30
    return T, n


@app.cell
def _(np):
    A = np.array([[-1, 0.4, 0.8], [1, 0, 0], [0, 1, 0]])
    b = np.array([[1, 0, 0.3]]).T
    return A, b


@app.cell(hide_code=True)
def _(mo, n, np):
    import wigglystuff

    x0_widget = mo.ui.anywidget(wigglystuff.Matrix(np.zeros((1, n))))
    xdes_widget = mo.ui.anywidget(wigglystuff.Matrix(np.array([[7, 2, -6]])))

    _a = mo.md(
        rf"""

        Choose a value for $x_0$ ...

        {x0_widget}
        """
    )

    _b = mo.md(
        rf"""
        ... and for $x_\text{{des}}$

        {xdes_widget}
        """
    )

    mo.hstack([_a, _b], justify="space-around")
    return x0_widget, xdes_widget


@app.cell
def _(x0_widget, xdes_widget):
    x0 = x0_widget.matrix
    xdes = xdes_widget.matrix
    return x0, xdes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Next, we specify the problem as a linear program using CVXPY.** This problem is linear because the objective and constraints are affine. (In fact, the objective is piecewise affine, but CVXPY rewrites it to be affine for you.)
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    return (cp,)


@app.cell
def _(A, T, b, cp, mo, n, x0, xdes):
    X, u = cp.Variable(shape=(n, T + 1)), cp.Variable(shape=(1, T))

    objective = cp.sum(cp.maximum(cp.abs(u), 2 * cp.abs(u) - 1))
    constraints = [
        X[:, 1:] == A @ X[:, :-1] + b @ u,
        X[:, 0] == x0,
        X[:, -1] == xdes,
    ]

    fuel_used = cp.Problem(cp.Minimize(objective), constraints).solve()
    mo.md(f"Achieved a fuel usage of {fuel_used:.02f}. 🚀")
    return (u,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Finally, we plot the chosen inputs over time.

    **🌊 Try it!** Change the initial and desired states; how do fuel usage and controls change? Can you explain what you see? You can also try experimenting with the value of $T$.
    """)
    return


@app.cell
def _(plot_solution, u):
    plot_solution(u)
    return


@app.cell
def _(T, cp, np):
    def plot_solution(u: cp.Variable):
        import matplotlib.pyplot as plt

        plt.step(np.arange(T), u.T.value)
        plt.axis("tight")
        plt.xlabel("$t$")
        plt.ylabel("$u$")
        return plt.gca()
    return (plot_solution,)


if __name__ == "__main__":
    app.run()

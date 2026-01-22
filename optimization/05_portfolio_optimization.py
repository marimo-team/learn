# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cvxpy==1.6.0",
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.2",
#     "scipy==1.15.1",
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
    # Portfolio optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this example we show how to use CVXPY to design a financial portfolio; this is called _portfolio optimization_.

    In portfolio optimization we have some amount of money to invest in any of $n$ different assets.
    We choose what fraction $w_i$ of our money to invest in each asset $i$, $i=1, \ldots, n$. The goal is to maximize return of the portfolio while minimizing risk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Asset returns and risk

    We will only model investments held for one period. The initial prices are $p_i > 0$. The end of period prices are $p_i^+ >0$. The asset (fractional) returns are $r_i = (p_i^+-p_i)/p_i$. The portfolio (fractional) return is $R = r^Tw$.

    A common model is that $r$ is a random variable with mean ${\bf E}r = \mu$ and covariance ${\bf E{(r-\mu)(r-\mu)^T}} = \Sigma$.
    It follows that $R$ is a random variable with ${\bf E}R = \mu^T w$ and ${\bf var}(R) = w^T\Sigma w$. In real-world applications, $\mu$ and $\Sigma$ are estimated from data and models, and $w$ is chosen using a library like CVXPY.

    ${\bf E}R$ is the (mean) *return* of the portfolio. ${\bf var}(R)$ is the *risk* of the portfolio. Portfolio optimization has two competing objectives: high return and low risk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classical (Markowitz) portfolio optimization

    Classical (Markowitz) portfolio optimization solves the optimization problem
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    \begin{array}{ll} \text{maximize} & \mu^T w - \gamma w^T\Sigma w\\
    \text{subject to} & {\bf 1}^T w = 1, w \geq 0,
    \end{array}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    where $w \in {\bf R}^n$ is the optimization variable and $\gamma >0$ is a constant called the *risk aversion parameter*. The constraint $\mathbf{1}^Tw = 1$ says the portfolio weight vector must sum to 1, and $w \geq 0$ says that we can't invest a negative amount into any asset.

    The objective $\mu^Tw - \gamma w^T\Sigma w$ is the *risk-adjusted return*. Varying $\gamma$ gives the optimal *risk-return trade-off*.
    We can get the same risk-return trade-off by fixing return and minimizing risk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example

    In the following code we compute and plot the optimal risk-return trade-off for $10$ assets. First we generate random problem data $\mu$ and $\Sigma$.
    """)
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell(hide_code=True)
def _(mo, np):
    import wigglystuff

    mu_widget = mo.ui.anywidget(
        wigglystuff.Matrix(
            np.array(
                [
                    [1.6],
                    [0.6],
                    [0.5],
                    [1.1],
                    [0.9],
                    [2.3],
                    [1.7],
                    [0.7],
                    [0.9],
                    [0.3],
                ]
            )
        )
    )


    mo.md(
        rf"""
        The value of $\mu$ is 

        {mu_widget.center()}

        _Try changing the entries of $\mu$ and see how the plots below change._
        """
    )
    return (mu_widget,)


@app.cell
def _(mu_widget, np):
    np.random.seed(1)
    n = 10
    mu = np.array(mu_widget.matrix)
    Sigma = np.random.randn(n, n)
    Sigma = Sigma.T.dot(Sigma)
    return Sigma, mu, n


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Next, we solve the problem for 100 different values of $\gamma$
    """)
    return


@app.cell
def _(Sigma, mu, n):
    import cvxpy as cp

    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret = mu.T @ w
    risk = cp.quad_form(w, Sigma)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w >= 0])
    return cp, gamma, prob, ret, risk


@app.cell
def _(cp, gamma, np, prob, ret, risk):
    _SAMPLES = 100
    risk_data = np.zeros(_SAMPLES)
    ret_data = np.zeros(_SAMPLES)
    gamma_vals = np.logspace(-2, 3, num=_SAMPLES)
    for _i in range(_SAMPLES):
        gamma.value = gamma_vals[_i]
        prob.solve()
        risk_data[_i] = cp.sqrt(risk).value
        ret_data[_i] = ret.value
    return gamma_vals, ret_data, risk_data


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Plotted below are the risk return tradeoffs for two values of $\gamma$ (blue squares), and the risk return tradeoffs for investing fully in each asset (red circles)
    """)
    return


@app.cell(hide_code=True)
def _(Sigma, cp, gamma_vals, mu, n, ret_data, risk_data):
    import matplotlib.pyplot as plt

    markers_on = [29, 40]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(risk_data, ret_data, "g-")
    for marker in markers_on:
        plt.plot(risk_data[marker], ret_data[marker], "bs")
        ax.annotate(
            "$\\gamma = %.2f$" % gamma_vals[marker],
            xy=(risk_data[marker] + 0.08, ret_data[marker] - 0.03),
        )
    for _i in range(n):
        plt.plot(cp.sqrt(Sigma[_i, _i]).value, mu[_i], "ro")
    plt.xlabel("Standard deviation")
    plt.ylabel("Return")
    plt.show()
    return markers_on, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We plot below the return distributions for the two risk aversion values marked on the trade-off curve.
    Notice that the probability of a loss is near 0 for the low risk value and far above 0 for the high risk value.
    """)
    return


@app.cell(hide_code=True)
def _(gamma, gamma_vals, markers_on, np, plt, prob, ret, risk):
    import scipy.stats as spstats

    plt.figure()
    for midx, _idx in enumerate(markers_on):
        gamma.value = gamma_vals[_idx]
        prob.solve()
        x = np.linspace(-2, 5, 1000)
        plt.plot(
            x,
            spstats.norm.pdf(x, ret.value, risk.value),
            label="$\\gamma = %.2f$" % gamma.value,
        )
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.show()
    return


if __name__ == "__main__":
    app.run()

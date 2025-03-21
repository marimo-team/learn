# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.4",
#     "scipy==1.15.2",
#     "altair==5.2.0",
#     "wigglystuff==0.1.10",
#     "pandas==2.2.3",
# ]
# ///

import marimo

__generated_with = "0.11.24"
app = marimo.App(width="medium", app_title="Binomial Distribution")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Binomial Distribution

        _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/binomial/), by Stanford professor Chris Piech._

        In this section, we will discuss the binomial distribution. To start, imagine the following example:

        Consider $n$ independent trials of an experiment where each trial is a "success" with probability $p$. Let $X$ be the number of successes in $n$ trials.

        This situation is truly common in the natural world, and as such, there has been a lot of research into such phenomena. Random variables like $X$ are called **binomial random variables**. If you can identify that a process fits this description, you can inherit many already proved properties such as the PMF formula, expectation, and variance!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Binomial Random Variable Definition

        $X \sim \text{Bin}(n, p)$ represents a binomial random variable where:

        - $X$ is our random variable (number of successes)
        - $\text{Bin}$ indicates it follows a binomial distribution
        - $n$ is the number of trials
        - $p$ is the probability of success in each trial

        ```
        X ~ Bin(n, p)
         ↑    ↑  ↑
         |    |  +-- Probability of
         |    |      success on each
         |    |      trial
         |    +-- Number of trials
         |
        Our random variable
          is distributed
          as a Binomial
        ```

        Here are a few examples of binomial random variables:

        - Number of heads in $n$ coin flips
        - Number of 1's in randomly generated length $n$ bit string
        - Number of disk drives crashed in 1000 computer cluster, assuming disks crash independently
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Properties of Binomial Distribution

        | Property | Formula |
        |----------|---------|
        | Notation | $X \sim \text{Bin}(n, p)$ |
        | Description | Number of "successes" in $n$ identical, independent experiments each with probability of success $p$ |
        | Parameters | $n \in \{0, 1, \dots\}$, the number of experiments<br>$p \in [0, 1]$, the probability that a single experiment gives a "success" |
        | Support | $x \in \{0, 1, \dots, n\}$ |
        | PMF equation | $P(X=x) = {n \choose x}p^x(1-p)^{n-x}$ |
        | Expectation | $E[X] = n \cdot p$ |
        | Variance | $\text{Var}(X) = n \cdot p \cdot (1-p)$ |

        Let's explore how the binomial distribution changes with different parameters.
        """
    )
    return


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    # Interactive elements using TangleSlider
    n_slider = mo.ui.anywidget(TangleSlider(
        amount=10, 
        min_value=1, 
        max_value=30, 
        step=1,
        digits=0,
        suffix=" trials"
    ))

    p_slider = mo.ui.anywidget(TangleSlider(
        amount=0.5, 
        min_value=0.01, 
        max_value=0.99, 
        step=0.01,
        digits=2,
        suffix=" probability"
    ))

    # Grid layout for the interactive controls
    controls = mo.vstack([
        mo.md("### Adjust Parameters to See How Binomial Distribution Changes"),
        mo.hstack([
            mo.md("**Number of trials (n):** "),
            n_slider
        ], justify="start"),
        mo.hstack([
            mo.md("**Probability of success (p):** "),
            p_slider
        ], justify="start"),
    ])
    return controls, n_slider, p_slider


@app.cell(hide_code=True)
def _(controls):
    controls
    return


@app.cell(hide_code=True)
def _(n_slider, np, p_slider, plt, stats):
    # Parameters from sliders
    _n = int(n_slider.amount)
    _p = p_slider.amount

    # Calculate PMF
    _x = np.arange(0, _n + 1)
    _pmf = stats.binom.pmf(_x, _n, _p)

    # Relevant stats
    _mean = _n * _p
    _variance = _n * _p * (1 - _p)
    _std_dev = np.sqrt(_variance)

    _fig, _ax = plt.subplots(figsize=(10, 6))

    # Plot PMF as bars
    _ax.bar(_x, _pmf, color='royalblue', alpha=0.7, label=f'PMF: P(X=k)')

    # Add a line
    _ax.plot(_x, _pmf, 'ro-', alpha=0.6, label='PMF line')

    # Add vertical lines
    _ax.axvline(x=_mean, color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {_mean:.2f}')

    # Shade the stdev region
    _ax.axvspan(_mean - _std_dev, _mean + _std_dev, alpha=0.2, color='green',
               label=f'±1 Std Dev: {_std_dev:.2f}')

    # Add labels and title
    _ax.set_xlabel('Number of Successes (k)')
    _ax.set_ylabel('Probability: P(X=k)')
    _ax.set_title(f'Binomial Distribution with n={_n}, p={_p:.2f}')

    # Annotations
    _ax.annotate(f'E[X] = {_mean:.2f}', 
                xy=(_mean, stats.binom.pmf(int(_mean), _n, _p)), 
                xytext=(_mean + 1, max(_pmf) * 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    _ax.annotate(f'Var(X) = {_variance:.2f}', 
                xy=(_mean, stats.binom.pmf(int(_mean), _n, _p) / 2), 
                xytext=(_mean + 1, max(_pmf) * 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # Grid and legend
    _ax.grid(alpha=0.3)
    _ax.legend()

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Relationship to Bernoulli Random Variables

        One way to think of the binomial is as the sum of $n$ Bernoulli variables. Say that $Y_i$ is an indicator Bernoulli random variable which is 1 if experiment $i$ is a success. Then if $X$ is the total number of successes in $n$ experiments, $X \sim \text{Bin}(n, p)$:

        $$X = \sum_{i=1}^n Y_i$$

        Recall that the outcome of $Y_i$ will be 1 or 0, so one way to think of $X$ is as the sum of those 1s and 0s.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Binomial Probability Mass Function (PMF)

        The most important property to know about a binomial is its [Probability Mass Function](https://marimo.app/https://github.com/marimo-team/learn/blob/main/probability/10_probability_mass_function.py):

        $$P(X=k) = {n \choose k}p^k(1-p)^{n-k}$$

        ```
        P(X = k) = (n) p^k(1-p)^(n-k)
         ↑           (k)
         |            ↑
         |            +-- Binomial coefficient:
         |                number of ways to choose
         |                k successes from n trials
         |
        Probability that our
        variable takes on the
        value k
        ```

        Recall, we derived this formula in Part 1. There is a complete example on the probability of $k$ heads in $n$ coin flips, where each flip is heads with probability $p$.

        To briefly review, if you think of each experiment as being distinct, then there are ${n \choose k}$ ways of permuting $k$ successes from $n$ experiments. For any of the mutually exclusive permutations, the probability of that permutation is $p^k \cdot (1-p)^{n-k}$.

        The name binomial comes from the term ${n \choose k}$ which is formally called the binomial coefficient.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Expectation of Binomial

        There is an easy way to calculate the expectation of a binomial and a hard way. The easy way is to leverage the fact that a binomial is the sum of Bernoulli indicator random variables $X = \sum_{i=1}^{n} Y_i$ where $Y_i$ is an indicator of whether the $i$-th experiment was a success: $Y_i \sim \text{Bernoulli}(p)$. 

        Since the [expectation of the sum](http://marimo.app/https://github.com/marimo-team/learn/blob/main/probability/11_expectation.py) of random variables is the sum of expectations, we can add the expectation, $E[Y_i] = p$, of each of the Bernoulli's:

        \begin{align}
        E[X] &= E\Big[\sum_{i=1}^{n} Y_i\Big] && \text{Since }X = \sum_{i=1}^{n} Y_i \\
        &= \sum_{i=1}^{n}E[ Y_i] && \text{Expectation of sum} \\
        &= \sum_{i=1}^{n}p && \text{Expectation of Bernoulli} \\
        &= n \cdot p && \text{Sum $n$ times}
        \end{align}

        The hard way is to use the definition of expectation:

        \begin{align}
        E[X] &= \sum_{i=0}^n i \cdot P(X = i) && \text{Def of expectation} \\
        &= \sum_{i=0}^n i \cdot {n \choose i} p^i(1-p)^{n-i} && \text{Sub in PMF} \\
        & \cdots && \text{Many steps later} \\
        &= n \cdot p
        \end{align}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Binomial Distribution in Python

        As you might expect, you can use binomial distributions in code. The standardized library for binomials is `scipy.stats.binom`.

        One of the most helpful methods that this package provides is a way to calculate the PMF. For example, say $n=5$, $p=0.6$ and you want to find $P(X=2)$, you could use the following code:
        """
    )
    return


@app.cell
def _(stats):
    # define variables for x, n, and p
    _n = 5  # Integer value for n
    _p = 0.6
    _x = 2

    # use scipy to compute the pmf
    p_x = stats.binom.pmf(_x, _n, _p)

    # use the probability for future work
    print(f'P(X = {_x}) = {p_x:.4f}')
    return (p_x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Another particularly helpful function is the ability to generate a random sample from a binomial. For example, say $X$ represents the number of requests to a website. We can draw 100 samples from this distribution using the following code:""")
    return


@app.cell
def _(n, p, stats):
    n_int = int(n)

    # samples from the binomial distribution
    samples = stats.binom.rvs(n_int, p, size=100)

    # Print the samples
    print(samples)
    return n_int, samples


@app.cell(hide_code=True)
def _(n_int, np, p, plt, samples, stats):
    # Plot histogram of samples
    plt.figure(figsize=(10, 5))
    plt.hist(samples, bins=np.arange(-0.5, n_int+1.5, 1), alpha=0.7, color='royalblue', 
             edgecolor='black', density=True)

    # Overlay the PMF
    x_values = np.arange(0, n_int+1)
    pmf_values = stats.binom.pmf(x_values, n_int, p)
    plt.plot(x_values, pmf_values, 'ro-', ms=8, label='Theoretical PMF')

    # Add labels and title
    plt.xlabel('Number of Successes')
    plt.ylabel('Relative Frequency / Probability')
    plt.title(f'Histogram of 100 Samples from Bin({n_int}, {p})')
    plt.legend()
    plt.grid(alpha=0.3)

    # Annotate
    plt.annotate('Sample mean: %.2f' % np.mean(samples), 
                xy=(0.7, 0.9), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    plt.annotate('Theoretical mean: %.2f' % (n_int*p), 
                xy=(0.7, 0.8), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.gca()
    return pmf_values, x_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You might be wondering what a random sample is! A random sample is a randomly chosen assignment for our random variable. Above we have 100 such assignments. The probability that value $k$ is chosen is given by the PMF: $P(X=k)$. 

        There are also functions for getting the mean, the variance, and more. You can read the [scipy.stats.binom documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html), especially the list of methods.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interactive Exploration of Binomial vs. Negative Binomial

        The standard binomial distribution is a special case of a broader family of distributions. One related distribution is the negative binomial, which can model count data with overdispersion (where the variance is larger than the mean).

        Below, you can explore how the negative binomial distribution compares to a Poisson distribution (which can be seen as a limiting case of the binomial as $n$ gets large and $p$ gets small, with $np$ held constant).

        Adjust the sliders to see how the parameters affect the distribution:

        *Note: The interactive visualization in this section was inspired by work from [liquidcarbon on GitHub](https://github.com/liquidcarbon).*
        """
    )
    return


@app.cell(hide_code=True)
def _(alpha_slider, chart, equation, mo, mu_slider):
    mo.vstack(
        [
            mo.md(f"## Negative Binomial Distribution (Poisson + Overdispersion)\n{equation}"),
            mo.hstack([mu_slider, alpha_slider], justify="start"),
            chart,
        ], justify='space-around'
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding
        Pick which of these statements about binomial distributions you think are correct:

        /// details | The variance of a binomial distribution is always equal to its mean
        ❌ Incorrect! The variance is $np(1-p)$ while the mean is $np$. They're only equal when $p=1$ (which is a degenerate case).
        ///

        /// details | If $X \sim \text{Bin}(n, p)$ and $Y \sim \text{Bin}(n, 1-p)$, then $X$ and $Y$ have the same variance
        ✅ Correct! $\text{Var}(X) = np(1-p)$ and $\text{Var}(Y) = n(1-p)p$, which are the same.
        ///

        /// details | As the number of trials increases, the binomial distribution approaches a normal distribution
        ✅ Correct! For large $n$, the binomial distribution can be approximated by a normal distribution with the same mean and variance.
        ///

        /// details | The PMF of a binomial distribution is symmetric when $p = 0.5$
        ✅ Correct! When $p = 0.5$, the PMF is symmetric around $n/2$.
        ///

        /// details | The sum of two independent binomial random variables with the same $p$ is also a binomial random variable
        ✅ Correct! If $X \sim \text{Bin}(n_1, p)$ and $Y \sim \text{Bin}(n_2, p)$ are independent, then $X + Y \sim \text{Bin}(n_1 + n_2, p)$.
        ///

        /// details | The maximum value of the PMF for $\text{Bin}(n,p)$ always occurs at $k = np$
        ❌ Incorrect! The mode (maximum value of PMF) is either $\lfloor (n+1)p \rfloor$ or $\lceil (n+1)p-1 \rceil$ depending on whether $(n+1)p$ is an integer.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        So we've explored the binomial distribution, and honestly, it's one of the most practical probability distributions you'll encounter. Think about it — anytime you're counting successes in a fixed number of trials (like those coin flips we discussed), this is your go-to distribution.

        I find it fascinating how the expectation is simply $np$. Such a clean, intuitive formula! And remember that neat visualization we saw earlier? When we adjusted the parameters, you could actually see how the distribution shape changes—becoming more symmetric as $n$ increases.

        The key things to take away:

        - The binomial distribution models the number of successes in $n$ independent trials, each with probability $p$ of success

        - Its PMF is given by the formula $P(X=k) = {n \choose k}p^k(1-p)^{n-k}$, which lets us calculate exactly how likely any specific number of successes is

        - The expected value is $E[X] = np$ and the variance is $Var(X) = np(1-p)$

        - It's related to other distributions: it's essentially a sum of Bernoulli random variables, and connects to both the negative binomial and Poisson distributions

        - In Python, the `scipy.stats.binom` module makes working with binomial distributions straightforward—you can generate random samples and calculate probabilities with just a few lines of code

        You'll see the binomial distribution pop up everywhere—from computer science to quality control, epidemiology, and data science. Any time you have scenarios with binary outcomes over multiple trials, this distribution has you covered.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Appendix code (helper functions, variables, etc.):""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import pandas as pd
    import altair as alt
    from wigglystuff import TangleSlider
    return TangleSlider, alt, np, pd, plt, stats


@app.cell(hide_code=True)
def _(mo):
    alpha_slider = mo.ui.slider(
        value=0.1,
        steps=[0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1],
        label="α (overdispersion)",
        show_value=True,
    )
    mu_slider = mo.ui.slider(
        value=100, start=1, stop=100, step=1, label="μ (mean)", show_value=True
    )
    return alpha_slider, mu_slider


@app.cell(hide_code=True)
def _():
    equation = """
    $$
    P(X = k) = \\frac{\\Gamma(k + \\frac{1}{\\alpha})}{\\Gamma(k + 1) \\Gamma(\\frac{1}{\\alpha})} \\left( \\frac{1}{\\mu \\alpha + 1} \\right)^{\\frac{1}{\\alpha}} \\left( \\frac{\\mu \\alpha}{\\mu \\alpha + 1} \\right)^k
    $$

    $$
    \\sigma^2 = \\mu + \\alpha \\mu^2
    $$
    """
    return (equation,)


@app.cell(hide_code=True)
def _(alpha_slider, alt, mu_slider, np, pd, stats):
    mu = mu_slider.value
    alpha = alpha_slider.value
    n = 1000 - mu if alpha == 0 else 1 / alpha
    p = n / (mu + n)
    x = np.arange(0, mu * 3 + 1, 1)
    df = pd.DataFrame(
        {
            "x": x,
            "y": stats.nbinom.pmf(x, n, p),
            "y_poi": stats.nbinom.pmf(x, 1000 - mu, 1 - mu / 1000),
        }
    )
    r1k = stats.nbinom.rvs(n, p, size=1000)
    df["in 95% CI"] = df["x"].between(*np.percentile(r1k, q=[2.5, 97.5]))
    base = alt.Chart(df)

    chart_poi = base.mark_bar(
        fillOpacity=0.25, width=100 / mu, fill="magenta"
    ).encode(
        x=alt.X("x").scale(domain=(-0.4, x.max() + 0.4), nice=False),
        y=alt.Y("y_poi").scale(domain=(0, df.y_poi.max() * 1.1)).title(None),
    )
    chart_nb = base.mark_bar(fillOpacity=0.75, width=100 / mu).encode(
        x="x",
        y="y",
        fill=alt.Fill("in 95% CI")
        .scale(domain=[False, True], range=["#aaa", "#7c7"])
        .legend(orient="bottom-right"),
    )

    chart = (chart_poi + chart_nb).configure_view(continuousWidth=450)
    return alpha, base, chart, chart_nb, chart_poi, df, mu, n, p, r1k, x


if __name__ == "__main__":
    app.run()

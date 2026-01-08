# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.3",
#     "scipy==1.15.2",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Random Variables")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    return np, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random Variables

    _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/rvs/), by Stanford professor Chris Piech._

    Random variables are functions that map outcomes from a probability space to numbers. This mathematical abstraction allows us to:

    - Work with numerical outcomes in probability
    - Calculate expected values and variances
    - Model real-world phenomena quantitatively
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Types of Random Variables

    ### Discrete Random Variables
    - Take on countable values (finite or infinite)
    - Described by a probability mass function (PMF)
    - Example: Number of heads in 3 coin flips

    ### Continuous Random Variables
    - Take on uncountable values in an interval
    - Described by a probability density function (PDF)
    - Example: Height of a randomly selected person
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Properties of Random Variables

    Each random variable has several key properties:

    | Property | Description | Example |
    |----------|-------------|---------|
    | Meaning | Semantic description | Number of successes in n trials |
    | Symbol | Notation used | $X$, $Y$, $Z$ |
    | Support/Range | Possible values | $\{0,1,2,...,n\}$ for binomial |
    | Distribution | PMF or PDF | $p_X(x)$ or $f_X(x)$ |
    | Expectation | Weighted average | $E[X]$ |
    | Variance | Measure of spread | $\text{Var}(X)$ |
    | Standard Deviation | Square root of variance | $\sigma_X$ |
    | Mode | Most likely value | argmax$_x$ $p_X(x)$ |

    Additional properties include:

    - [Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) (measure of uncertainty)
    - [Median](https://en.wikipedia.org/wiki/Median) (middle value)
    - [Skewness](https://en.wikipedia.org/wiki/Skewness) (asymmetry measure)
    - [Kurtosis](https://en.wikipedia.org/wiki/Kurtosis) (tail heaviness measure)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probability Mass Functions (PMF)

    For discrete random variables, the PMF $p_X(x)$ gives the probability that $X$ equals $x$:

    $p_X(x) = P(X = x)$

    Properties of a PMF:

    1. $p_X(x) \geq 0$ for all $x$
    2. $\sum_x p_X(x) = 1$

    Let's implement a PMF for rolling a fair die:
    """)
    return


@app.cell
def _(np, plt):
    def die_pmf(x):
        if x in [1, 2, 3, 4, 5, 6]:
            return 1 / 6
        return 0

    # Plot the PMF
    _x = np.arange(1, 7)
    probabilities = [die_pmf(i) for i in _x]

    plt.figure(figsize=(8, 2))
    plt.bar(_x, probabilities)
    plt.title("PMF of Rolling a Fair Die")
    plt.xlabel("Outcome")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probability Density Functions (PDF)

    For continuous random variables, we use a PDF $f_X(x)$. The probability of $X$ falling in an interval $[a,b]$ is:

    $P(a \leq X \leq b) = \int_a^b f_X(x)dx$

    Properties of a PDF:

    1. $f_X(x) \geq 0$ for all $x$
    2. $\int_{-\infty}^{\infty} f_X(x)dx = 1$

    Let's look at the normal distribution, a common continuous random variable:
    """)
    return


@app.cell
def _(np, plt, stats):
    # Generate points for plotting
    _x = np.linspace(-4, 4, 100)
    _pdf = stats.norm.pdf(_x, loc=0, scale=1)

    plt.figure(figsize=(8, 4))
    plt.plot(_x, _pdf, "b-", label="PDF")
    plt.fill_between(_x, _pdf, where=(_x >= -1) & (_x <= 1), alpha=0.3)
    plt.title("Standard Normal Distribution")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expected Value

    The expected value $E[X]$ is the long-run average of a random variable.

    For discrete random variables:
    $E[X] = \sum_x x \cdot p_X(x)$

    For continuous random variables:
    $E[X] = \int_{-\infty}^{\infty} x \cdot f_X(x)dx$

    Properties:

    1. $E[aX + b] = aE[X] + b$
    2. $E[X + Y] = E[X] + E[Y]$
    """)
    return


@app.cell
def _(np):
    def expected_value_discrete(x_values, probabilities):
        return sum(x * p for x, p in zip(x_values, probabilities))

    # Example: Expected value of a fair die roll
    die_values = np.arange(1, 7)
    die_probs = np.ones(6) / 6

    E_X = expected_value_discrete(die_values, die_probs)
    return E_X, die_probs, die_values


@app.cell
def _(E_X):
    E_X
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variance

    The variance $\text{Var}(X)$ measures the spread of a random variable around its mean:

    $\text{Var}(X) = E[(X - E[X])^2]$

    This can be computed as:
    $\text{Var}(X) = E[X^2] - (E[X])^2$

    Properties:

    1. $\text{Var}(aX) = a^2Var(X)$
    2. $\text{Var}(X + b) = Var(X)$
    """)
    return


@app.cell
def _(E_X, die_probs, die_values, np):
    def variance_discrete(x_values, probabilities, expected_value):
        squared_diff = [(x - expected_value) ** 2 for x in x_values]
        return sum(d * p for d, p in zip(squared_diff, probabilities))

    # Example: Variance of a fair die roll
    var_X = variance_discrete(die_values, die_probs, E_X)
    std_X = np.sqrt(var_X)
    return std_X, var_X, variance_discrete


@app.cell(hide_code=True)
def _(mo, std_X, var_X):
    mo.md(
        f"""
        ### Examples of Variance Calculation

        For our fair die example:

        - Variance: {var_X:.2f}
        - Standard Deviation: {std_X:.2f}

        This means that on average, a roll deviates from the mean (3.5) by about {std_X:.2f} units.

        Let's look another example for a fair coin:
        """
    )
    return


@app.cell
def _(variance_discrete):
    # Fair coin (X = 0 or 1)
    coin_values = [0, 1]
    coin_probs = [0.5, 0.5]
    coin_mean = sum(x * p for x, p in zip(coin_values, coin_probs))
    coin_var = variance_discrete(coin_values, coin_probs, coin_mean)
    return (coin_var,)


@app.cell
def _(np, stats, variance_discrete):
    # Standard normal (discretized for example)
    normal_values = np.linspace(-3, 3, 100)
    normal_probs = stats.norm.pdf(normal_values)
    normal_probs = normal_probs / sum(normal_probs)  # normalize
    normal_mean = 0
    normal_var = variance_discrete(normal_values, normal_probs, normal_mean)
    return (normal_var,)


@app.cell
def _(np, variance_discrete):
    # Uniform on [0,1] (discretized for example)
    uniform_values = np.linspace(0, 1, 100)
    uniform_probs = np.ones_like(uniform_values) / len(uniform_values)
    uniform_mean = 0.5
    uniform_var = variance_discrete(uniform_values, uniform_probs, uniform_mean)
    return (uniform_var,)


@app.cell(hide_code=True)
def _(coin_var, mo, normal_var, uniform_var):
    mo.md(
        rf"""
        Let's look at some calculated variances:

        - Fair coin (X = 0 or 1): $\text{{Var}}(X) = {coin_var:.4f}$
        - Standard normal distribution (discretized): $\text{{Var(X)}} ≈ {normal_var:.4f}$
        - Uniform distribution on $[0,1]$ (discretized): $\text{{Var(X)}} ≈ {uniform_var:.4f}$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Common Distributions

    1. Bernoulli Distribution
        - Models a single success/failure experiment
        - $P(X = 1) = p$, $P(X = 0) = 1-p$
        - $E[X] = p$, $\text{Var}(X) = p(1-p)$

    2. Binomial Distribution

        - Models number of successes in $n$ independent trials
        - $P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$
        - $E[X] = np$, $\text{Var}(X) = np(1-p)$

    3. Normal Distribution

        - Bell-shaped curve defined by mean $\mu$ and variance $\sigma^2$
        - PDF: $f_X(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
        - $E[X] = \mu$, $\text{Var}(X) = \sigma^2$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Comparing Discrete and Continuous Distributions

    This example shows the relationship between a Binomial distribution (discrete) and its Normal approximation (continuous).
    The parameters control both distributions:

    - **Number of Trials**: Controls the range of possible values and the shape's width
    - **Success Probability**: Affects the distribution's center and skewness
    """)
    return


@app.cell
def _(mo, n_trials, p_success):
    mo.hstack([n_trials, p_success], justify="space-around")
    return


@app.cell(hide_code=True)
def _(mo):
    # Distribution parameters
    n_trials = mo.ui.slider(1, 20, value=10, label="Number of Trials")
    p_success = mo.ui.slider(0, 1, value=0.5, step=0.05, label="Success Probability")
    return n_trials, p_success


@app.cell(hide_code=True)
def _(n_trials, np, p_success, plt, stats):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    # Discrete: Binomial PMF
    k = np.arange(0, n_trials.value + 1)
    pmf = stats.binom.pmf(k, n_trials.value, p_success.value)
    ax1.bar(k, pmf, alpha=0.8, color="#1f77b4", label="PMF")
    ax1.set_title(f"Binomial PMF (n={n_trials.value}, p={p_success.value})")
    ax1.set_xlabel("Number of Successes")
    ax1.set_ylabel("Probability")
    ax1.grid(True, alpha=0.3)

    # Continuous: Normal PDF approx.
    mu = n_trials.value * p_success.value
    sigma = np.sqrt(n_trials.value * p_success.value * (1 - p_success.value))
    x = np.linspace(max(0, mu - 4 * sigma), min(n_trials.value, mu + 4 * sigma), 100)
    pdf = stats.norm.pdf(x, mu, sigma)

    ax2.plot(x, pdf, "r-", linewidth=2, label="PDF")
    ax2.fill_between(x, pdf, alpha=0.3, color="red")
    ax2.set_title(f"Normal PDF (μ={mu:.1f}, σ={sigma:.1f})")
    ax2.set_xlabel("Continuous Approximation")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)

    # Set consistent x-axis limits for better comparison
    ax1.set_xlim(-0.5, n_trials.value + 0.5)
    ax2.set_xlim(-0.5, n_trials.value + 0.5)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo, n_trials, np, p_success):
    mo.md(f"""
    **Current Distribution Properties:**

    - Mean (μ) = {n_trials.value * p_success.value:.2f}
    - Standard Deviation (σ) = {np.sqrt(n_trials.value * p_success.value * (1 - p_success.value)):.2f}

    Notice how the Normal distribution (right) approximates the Binomial distribution (left) better when:

    1. The number of trials is larger
    2. The success probability is closer to 0.5
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practice Problems

    ### Problem 1: Discrete Random Variable
    Let $X$ be the sum when rolling two fair dice. Find:

    1. The support of $X$
    2. The PMF $p_X(x)$
    3. $E[X]$ and $\text{Var}(X)$

    <details>
    <summary>Solution</summary>
    Let's solve this step by step:
    ```python
    def two_dice_pmf(x):
        outcomes = [(i,j) for i in range(1,7) for j in range(1,7)]
        favorable = [pair for pair in outcomes if sum(pair) == x]
        return len(favorable)/36

    # Support: {2,3,...,12}
    # E[X] = 7
    # Var(X) = 5.83
    ```
    </details>

    ### Problem 2: Continuous Random Variable
    For a uniform random variable on $[0,1]$, verify that:

    1. The PDF integrates to 1
    2. $E[X] = 1/2$
    3. $\text{Var}(X) = 1/12$

    Try solving this yourself first, then check the solution below.
    """)
    return


@app.cell
def _():
    # DIY
    return


@app.cell(hide_code=True)
def _(mktext, mo):
    mo.accordion({"Solution": mktext}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mktext=mo.md(r"""
    Let's solve each part:

    1. **PDF integrates to 1**:
       $\int_0^1 1 \, dx = [x]_0^1 = 1 - 0 = 1$

    2. **Expected Value**:
       $E[X] = \int_0^1 x \cdot 1 \, dx = [\frac{x^2}{2}]_0^1 = \frac{1}{2} - 0 = \frac{1}{2}$

    3. **Variance**:
       $\text{Var}(X) = E[X^2] - (E[X])^2$

       First calculate $E[X^2]$:
       $E[X^2] = \int_0^1 x^2 \cdot 1 \, dx = [\frac{x^3}{3}]_0^1 = \frac{1}{3}$

       Then:
       $\text{Var}(X) = \frac{1}{3} - (\frac{1}{2})^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}$
    """)
    return (mktext,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤔 Test Your Understanding

    Pick which of these statements about random variables you think are correct:

    <details>
    <summary>The probability density function can be greater than 1</summary>
    ✅ Correct! Unlike PMFs, PDFs can exceed 1 as long as the total area equals 1.
    </details>

    <details>
    <summary>The expected value of a random variable must equal one of its possible values</summary>
    ❌ Incorrect! For example, the expected value of a fair die is 3.5, which is not a possible outcome.
    </details>

    <details>
    <summary>Adding a constant to a random variable changes its variance</summary>
    ❌ Incorrect! Adding a constant shifts the distribution but doesn't affect its spread.
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    You've learned:

    - The difference between discrete and continuous random variables
    - How PMFs and PDFs describe probability distributions
    - Methods for calculating expected values and variances
    - Properties of common probability distributions

    In the next lesson, we'll explore Probability Mass Functions in detail, focusing on their properties and applications.
    """)
    return


if __name__ == "__main__":
    app.run()

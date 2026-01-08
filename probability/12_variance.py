# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.3",
#     "scipy==1.15.2",
#     "wigglystuff==0.1.10",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Variance")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Variance

    _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/variance/), by Stanford professor Chris Piech._

    In our previous exploration of random variables, we learned about expectation - a measure of central tendency. However, knowing the average value alone doesn't tell us everything about a distribution. Consider these questions:

    - How spread out are the values around the mean?
    - How reliable is the expectation as a predictor of individual outcomes?
    - How much do individual samples typically deviate from the average?

    This is where **variance** comes in - it measures the spread or dispersion of a random variable around its expected value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Definition of Variance

    The variance of a random variable $X$ with expected value $\mu = E[X]$ is defined as:

    $$\text{Var}(X) = E[(X-\mu)^2]$$

    This definition captures the average squared deviation from the mean. There's also an equivalent, often more convenient formula:

    $$\text{Var}(X) = E[X^2] - (E[X])^2$$

    /// tip
    The second formula is usually easier to compute, as it only requires calculating $E[X^2]$ and $E[X]$, rather than working with deviations from the mean.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Intuition Through Example

    Let's look at a real-world example that illustrates why variance is important. Consider three different groups of graders evaluating assignments in a massive online course. Each grader has their own "grading distribution" - their pattern of assigning scores to work that deserves a 70/100.

    The visualization below shows the probability distributions for three types of graders. Try clicking and dragging the blue numbers to adjust the parameters and see how they affect the variance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// TIP
    Try adjusting the blue numbers above to see how:

    - Increasing spread increases variance
    - The mixture ratio affects how many outliers appear in Grader C's distribution
    - Changing the true grade shifts all distributions but maintains their relative variances
    """)
    return


@app.cell(hide_code=True)
def _(controls):
    controls
    return


@app.cell(hide_code=True)
def _(
    grader_a_spread,
    grader_b_spread,
    grader_c_mix,
    np,
    plt,
    stats,
    true_grade,
):
    # Create data for three grader distributions
    _grader_x = np.linspace(40, 100, 200)

    # Calculate actual variances
    var_a = grader_a_spread.amount**2
    var_b = grader_b_spread.amount**2
    var_c = (1-grader_c_mix.amount) * 3**2 + grader_c_mix.amount * 8**2 + \
            grader_c_mix.amount * (1-grader_c_mix.amount) * (8-3)**2  # Mixture variance formula

    # Grader A: Wide spread around true grade
    grader_a = stats.norm.pdf(_grader_x, loc=true_grade.amount, scale=grader_a_spread.amount)

    # Grader B: Narrow spread around true grade
    grader_b = stats.norm.pdf(_grader_x, loc=true_grade.amount, scale=grader_b_spread.amount)

    # Grader C: Mixture of distributions
    grader_c = (1-grader_c_mix.amount) * stats.norm.pdf(_grader_x, loc=true_grade.amount, scale=3) + \
               grader_c_mix.amount * stats.norm.pdf(_grader_x, loc=true_grade.amount, scale=8)

    grader_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each distribution
    ax1.fill_between(_grader_x, grader_a, alpha=0.3, color='green', label=f'Var ≈ {var_a:.2f}')
    ax1.axvline(x=true_grade.amount, color='black', linestyle='--', label='True Grade')
    ax1.set_title('Grader A: High Variance')
    ax1.set_xlabel('Grade')
    ax1.set_ylabel('Pr(G = g)')
    ax1.set_ylim(0, max(grader_a)*1.1)

    ax2.fill_between(_grader_x, grader_b, alpha=0.3, color='blue', label=f'Var ≈ {var_b:.2f}')
    ax2.axvline(x=true_grade.amount, color='black', linestyle='--')
    ax2.set_title('Grader B: Low Variance')
    ax2.set_xlabel('Grade')
    ax2.set_ylim(0, max(grader_b)*1.1)

    ax3.fill_between(_grader_x, grader_c, alpha=0.3, color='purple', label=f'Var ≈ {var_c:.2f}')
    ax3.axvline(x=true_grade.amount, color='black', linestyle='--')
    ax3.set_title('Grader C: Mixed Distribution')
    ax3.set_xlabel('Grade')
    ax3.set_ylim(0, max(grader_c)*1.1)

    # Add annotations to explain what's happening
    ax1.annotate('Wide spread = high variance', 
                xy=(true_grade.amount, max(grader_a)*0.5),
                xytext=(true_grade.amount-15, max(grader_a)*0.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    ax2.annotate('Narrow spread = low variance', 
                xy=(true_grade.amount, max(grader_b)*0.5),
                xytext=(true_grade.amount+8, max(grader_b)*0.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    ax3.annotate('Mixture creates outliers', 
                xy=(true_grade.amount+15, grader_c[np.where(_grader_x >= true_grade.amount+15)[0][0]]),
                xytext=(true_grade.amount+5, max(grader_c)*0.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # Add legends and adjust layout
    for _ax in [ax1, ax2, ax3]:
        _ax.legend()
        _ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note
    All three distributions have the same expected value (the true grade), but they differ significantly in their spread:

    - **Grader A** has high variance - grades vary widely from the true value
    - **Grader B** has low variance - grades consistently stay close to the true value
    - **Grader C** has a mixture distribution - mostly consistent but with occasional extreme values

    This illustrates why variance is crucial: two distributions can have the same mean but behave very differently in practice.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computing Variance

    Let's work through some concrete examples to understand how to calculate variance.

    ### Example 1: Fair Die Roll

    Consider rolling a fair six-sided die. We'll calculate its variance step by step:
    """)
    return


@app.cell
def _(np):
    # Define the die values and probabilities
    die_values = np.array([1, 2, 3, 4, 5, 6])
    die_probs = np.array([1/6] * 6)

    # Calculate E[X]
    expected_value = np.sum(die_values * die_probs)

    # Calculate E[X^2]
    expected_square = np.sum(die_values**2 * die_probs)

    # Calculate Var(X) = E[X^2] - (E[X])^2
    variance = expected_square - expected_value**2

    # Calculate standard deviation
    std_dev = np.sqrt(variance)

    print(f"E[X] = {expected_value:.2f}")
    print(f"E[X^2] = {expected_square:.2f}")
    print(f"Var(X) = {variance:.2f}")
    print(f"Standard Deviation = {std_dev:.2f}")
    return die_probs, die_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// NOTE
    For a fair die:

    - The expected value (3.50) tells us the average roll
    - The variance (2.92) tells us how much typical rolls deviate from this average
    - The standard deviation (1.71) gives us this spread in the original units
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Properties of Variance

    Variance has several important properties that make it useful for analyzing random variables:

    1. **Non-negativity**: $\text{Var}(X) \geq 0$ for any random variable $X$
    2. **Variance of a constant**: $\text{Var}(c) = 0$ for any constant $c$
    3. **Scaling**: $\text{Var}(aX) = a^2\text{Var}(X)$ for any constant $a$
    4. **Translation**: $\text{Var}(X + b) = \text{Var}(X)$ for any constant $b$
    5. **Independence**: If $X$ and $Y$ are independent, then $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

    Let's verify a property with an example.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Proof of Variance Formula

    The equivalence of the two variance formulas is a fundamental result in probability theory. Here's the proof:

    Starting with the definition $\text{Var}(X) = E[(X-\mu)^2]$ where $\mu = E[X]$:

    \begin{align}
    \text{Var}(X) &= E[(X-\mu)^2] \\
    &= \sum_x(x-\mu)^2P(x) && \text{Definition of Expectation}\\
    &= \sum_x (x^2 -2\mu x + \mu^2)P(x) && \text{Expanding the square}\\
    &= \sum_x x^2P(x)- 2\mu \sum_x xP(x) + \mu^2 \sum_x P(x) && \text{Distributing the sum}\\
    &= E[X^2]- 2\mu E[X] + \mu^2 && \text{Definition of expectation}\\
    &= E[X^2]- 2(E[X])^2 + (E[X])^2 && \text{Since }\mu = E[X]\\
    &= E[X^2]- (E[X])^2 && \text{Simplifying}
    \end{align}

    /// tip
    This proof shows why the formula $\text{Var}(X) = E[X^2] - (E[X])^2$ is so useful - it's much easier to compute $E[X^2]$ and $E[X]$ separately than to work with deviations directly.
    """)
    return


@app.cell
def _(die_probs, die_values, np):
    # Demonstrate scaling property
    a = 2  # Scale factor

    # Original variance
    original_var = np.sum(die_values**2 * die_probs) - (np.sum(die_values * die_probs))**2

    # Scaled random variable variance
    scaled_values = a * die_values
    scaled_var = np.sum(scaled_values**2 * die_probs) - (np.sum(scaled_values * die_probs))**2

    print(f"Original Variance: {original_var:.2f}")
    print(f"Scaled Variance (a={a}): {scaled_var:.2f}")
    print(f"a^2 * Original Variance: {a**2 * original_var:.2f}")
    print(f"Property holds: {abs(scaled_var - a**2 * original_var) < 1e-10}")
    return


@app.cell
def _():
    # DIY : Prove more properties as shown above
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Standard Deviation

    While variance is mathematically convenient, it has one practical drawback: its units are squared. For example, if we're measuring grades (0-100), the variance is in "grade points squared." This makes it hard to interpret intuitively.

    The **standard deviation**, denoted by $\sigma$ or $\text{SD}(X)$, is the square root of variance:

    $$\sigma = \sqrt{\text{Var}(X)}$$

    /// tip
    Standard deviation is often more intuitive because it's in the same units as the original data. For a normal distribution, approximately:
    - 68% of values fall within 1 standard deviation of the mean
    - 95% of values fall within 2 standard deviations
    - 99.7% of values fall within 3 standard deviations
    """)
    return


@app.cell(hide_code=True)
def _(controls1):
    controls1
    return


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    normal_mean = mo.ui.anywidget(TangleSlider(
        amount=0, 
        min_value=-5, 
        max_value=5, 
        step=0.5,
        digits=1,
        suffix=" units"
    ))

    normal_std = mo.ui.anywidget(TangleSlider(
        amount=1, 
        min_value=0.1, 
        max_value=3, 
        step=0.1,
        digits=1,
        suffix=" units"
    ))

    # Create a grid layout for the controls
    controls1 = mo.vstack([
        mo.md("### Interactive Normal Distribution"),
        mo.hstack([
            mo.md("Adjust the parameters to see how standard deviation affects the shape of the distribution:"),
        ]),
        mo.hstack([
            mo.md("Mean (μ): "),
            normal_mean,
            mo.md("   Standard deviation (σ): "),
            normal_std
        ], justify="start"),
    ])
    return controls1, normal_mean, normal_std


@app.cell(hide_code=True)
def _(normal_mean, normal_std, np, plt, stats):
    # data for normal distribution
    _normal_x = np.linspace(-10, 10, 1000)
    _normal_y = stats.norm.pdf(_normal_x, loc=normal_mean.amount, scale=normal_std.amount)

    # ranges for standard deviation intervals
    one_sigma_left = normal_mean.amount - normal_std.amount
    one_sigma_right = normal_mean.amount + normal_std.amount
    two_sigma_left = normal_mean.amount - 2 * normal_std.amount
    two_sigma_right = normal_mean.amount + 2 * normal_std.amount
    three_sigma_left = normal_mean.amount - 3 * normal_std.amount
    three_sigma_right = normal_mean.amount + 3 * normal_std.amount

    # Create the plot
    normal_fig, normal_ax = plt.subplots(figsize=(10, 6))

    # Plot the distribution
    normal_ax.plot(_normal_x, _normal_y, 'b-', linewidth=2)

    # stdev intervals
    normal_ax.fill_between(_normal_x, 0, _normal_y, where=(_normal_x >= one_sigma_left) & (_normal_x <= one_sigma_right), 
                   alpha=0.3, color='red', label='68% (±1σ)')
    normal_ax.fill_between(_normal_x, 0, _normal_y, where=(_normal_x >= two_sigma_left) & (_normal_x <= two_sigma_right), 
                   alpha=0.2, color='green', label='95% (±2σ)')
    normal_ax.fill_between(_normal_x, 0, _normal_y, where=(_normal_x >= three_sigma_left) & (_normal_x <= three_sigma_right), 
                   alpha=0.1, color='blue', label='99.7% (±3σ)')

    # vertical lines for the mean and standard deviations
    normal_ax.axvline(x=normal_mean.amount, color='black', linestyle='-', linewidth=1.5, label='Mean (μ)')
    normal_ax.axvline(x=one_sigma_left, color='red', linestyle='--', linewidth=1)
    normal_ax.axvline(x=one_sigma_right, color='red', linestyle='--', linewidth=1)
    normal_ax.axvline(x=two_sigma_left, color='green', linestyle='--', linewidth=1)
    normal_ax.axvline(x=two_sigma_right, color='green', linestyle='--', linewidth=1)

    # annotations
    normal_ax.annotate(f'μ = {normal_mean.amount:.2f}', 
               xy=(normal_mean.amount, max(_normal_y)*0.5),
               xytext=(normal_mean.amount + 0.5, max(_normal_y)*0.8),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    normal_ax.annotate(f'σ = {normal_std.amount:.2f}', 
               xy=(one_sigma_right, stats.norm.pdf(one_sigma_right, loc=normal_mean.amount, scale=normal_std.amount)),
               xytext=(one_sigma_right + 0.5, max(_normal_y)*0.6),
               arrowprops=dict(facecolor='red', shrink=0.05, width=1))

    # labels and title
    normal_ax.set_xlabel('Value')
    normal_ax.set_ylabel('Probability Density')
    normal_ax.set_title(f'Normal Distribution with μ = {normal_mean.amount:.2f} and σ = {normal_std.amount:.2f}')

    # legend and grid
    normal_ax.legend()
    normal_ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip
    The interactive visualization above demonstrates how standard deviation (σ) affects the shape of a normal distribution:

    - The **red region** covers μ ± 1σ, containing approximately 68% of the probability
    - The **green region** covers μ ± 2σ, containing approximately 95% of the probability
    - The **blue region** covers μ ± 3σ, containing approximately 99.7% of the probability

    This is known as the "68-95-99.7 rule" or the "empirical rule" and is a useful heuristic for understanding the spread of data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤔 Test Your Understanding

    Choose what you believe are the correct options in the questions below:

    <details>
    <summary>The variance of a random variable can be negative.</summary>
    ❌ False! Variance is defined as an expected value of squared deviations, and squares are always non-negative.
    </details>

    <details>
    <summary>If X and Y are independent random variables, then Var(X + Y) = Var(X) + Var(Y).</summary>
    ✅ True! This is one of the key properties of variance for independent random variables.
    </details>

    <details>
    <summary>Multiplying a random variable by 2 multiplies its variance by 2.</summary>
    ❌ False! Multiplying a random variable by a constant a multiplies its variance by a². So multiplying by 2 multiplies variance by 4.
    </details>

    <details>
    <summary>Standard deviation is always equal to the square root of variance.</summary>
    ✅ True! By definition, standard deviation σ = √Var(X).
    </details>

    <details>
    <summary>If Var(X) = 0, then X must be a constant.</summary>
    ✅ True! Zero variance means there is no spread around the mean, so X can only take one value.
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    Variance gives us a way to measure how spread out a random variable is around its mean. It's like the "uncertainty" in our expectation - a high variance means individual outcomes can differ widely from what we expect on average.

    Standard deviation brings this measure back to the original units, making it easier to interpret. For grades, a standard deviation of 10 points means typical grades fall within about 10 points of the average.

    Variance pops up everywhere - from weather forecasts (how reliable is the predicted temperature?) to financial investments (how risky is this stock?) to quality control (how consistent is our manufacturing process?).

    In our next notebook, we'll explore more properties of random variables and see how they combine to form more complex distributions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Appendix (containing helper code):
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    from wigglystuff import TangleSlider
    return TangleSlider, np, plt, stats


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    # Create interactive elements using TangleSlider for a more inline experience
    true_grade = mo.ui.anywidget(TangleSlider(
        amount=70, 
        min_value=50, 
        max_value=90, 
        step=5,
        digits=0,
        suffix=" points"
    ))

    grader_a_spread = mo.ui.anywidget(TangleSlider(
        amount=10, 
        min_value=5, 
        max_value=20, 
        step=1,
        digits=0,
        suffix=" points"
    ))

    grader_b_spread = mo.ui.anywidget(TangleSlider(
        amount=2, 
        min_value=1, 
        max_value=5, 
        step=0.5,
        digits=1,
        suffix=" points"
    ))

    grader_c_mix = mo.ui.anywidget(TangleSlider(
        amount=0.2, 
        min_value=0, 
        max_value=1, 
        step=0.05,
        digits=2,
        suffix=" proportion"
    ))
    return grader_a_spread, grader_b_spread, grader_c_mix, true_grade


@app.cell(hide_code=True)
def _(grader_a_spread, grader_b_spread, grader_c_mix, mo, true_grade):
    # Create a grid layout for the interactive controls
    controls = mo.vstack([
        mo.md("### Adjust Parameters to See How Variance Changes"),
        mo.hstack([
            mo.md("**True grade:** The correct score that should be assigned is "),
            true_grade,
            mo.md(" out of 100.")
        ], justify="start"),
        mo.hstack([
            mo.md("**Grader A:** Has a wide spread with standard deviation of "),
            grader_a_spread,
            mo.md(" points.")
        ], justify="start"),
        mo.hstack([
            mo.md("**Grader B:** Has a narrow spread with standard deviation of "),
            grader_b_spread,
            mo.md(" points.")
        ], justify="start"),
        mo.hstack([
            mo.md("**Grader C:** Has a mixture distribution with "),
            grader_c_mix,
            mo.md(" proportion of outliers.")
        ], justify="start"),
    ])
    return (controls,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "scipy==1.15.2",
#     "wigglystuff==0.1.10",
#     "numpy==2.2.4",
# ]
# ///

import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium", app_title="Normal Distribution")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Normal Distribution

        _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/normal/), by Stanford professor Chris Piech._

        The Normal (also known as Gaussian) distribution is one of the most important probability distributions in statistics and data science. It's characterized by a symmetric bell-shaped curve and is fully defined by two parameters: mean (μ) and variance (σ²).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Normal Random Variable Definition

        The Normal (or Gaussian) random variable is denoted as:

        $$X \sim \mathcal{N}(\mu, \sigma^2)$$

        Where:

        - $X$ is our random variable
        - $\mathcal{N}$ indicates it follows a Normal distribution
        - $\mu$ is the mean parameter
        - $\sigma^2$ is the variance parameter (sometimes written as $\sigma$ for standard deviation)

        ```
        X ~ N(μ, σ²)
         ↑   ↑  ↑  ↑
         |   |  |  +-- Variance (spread)
         |   |  |      of the distribution
         |   |  +-- Mean (center)
         |   |      of the distribution
         |   +-- Indicates Normal
         |      distribution
         |
        Our random variable
        ```

        The Normal distribution is particularly important for many reasons:

        1. It arises naturally from the sum of independent random variables (Central Limit Theorem)
        2. It appears frequently in natural phenomena
        3. It is the maximum entropy distribution given a fixed mean and variance
        4. It simplifies many mathematical calculations in statistics and probability
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Properties of Normal Distribution

        | Property | Formula |
        |----------|---------|
        | Notation | $X \sim \mathcal{N}(\mu, \sigma^2)$ |
        | Description | A common, naturally occurring distribution |
        | Parameters | $\mu \in \mathbb{R}$, the mean<br>$\sigma^2 \in \mathbb{R}^+$, the variance |
        | Support | $x \in \mathbb{R}$ |
        | PDF equation | $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$ |
        | CDF equation | $F(x) = \Phi(\frac{x-\mu}{\sigma})$ where $\Phi$ is the CDF of the standard normal |
        | Expectation | $E[X] = \mu$ |
        | Variance | $\text{Var}(X) = \sigma^2$ |

        The PDF (Probability Density Function) reaches its maximum value at $x = \mu$, where the exponent becomes zero and $e^0 = 1$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mean_slider, mo, std_slider):
    mo.md(
        f"""
        The figure below shows a comparison between:

        - The **Standard Normal Distribution** (purple curve): N(0, 1)
        - A **Normal Distribution** with the parameters you selected (blue curve)

        Adjust the mean (μ) {mean_slider} and standard deviation (σ) {std_slider} below to see how the normal distribution changes shape.

        """
    )
    return


@app.cell(hide_code=True)
def _(
    create_distribution_comparison,
    fig_to_image,
    mean_slider,
    mo,
    std_slider,
):
    # values from the sliders
    current_mu = mean_slider.amount
    current_sigma = std_slider.amount

    # Create plot
    comparison_fig = create_distribution_comparison(current_mu, current_sigma)

    # Call, convert and display
    comp_image = mo.image(fig_to_image(comparison_fig), width="100%")
    comp_image
    return comp_image, comparison_fig, current_mu, current_sigma


@app.cell(hide_code=True)
def _(mean_slider, mo, std_slider):
    mo.md(
        f"""
        ## Interactive Normal Distribution Visualization

            The shape of a normal distribution is determined by two key parameters:

        - The **mean (μ):** {mean_slider} controls the center of the distribution.

        - The **standard deviation (σ):** {std_slider} controls the spread (width) of the distribution.

        Try adjusting these parameters to see how they affect the shape of the distribution below:

        """
    )
    return


@app.cell(hide_code=True)
def _(create_normal_pdf_plot, fig_to_image, mean_slider, mo, std_slider):
    # value from widgets
    _current_mu = mean_slider.amount
    _current_sigma = std_slider.amount

    # Create visualization
    pdf_fig = create_normal_pdf_plot(_current_mu, _current_sigma)

    # Display plot
    pdf_image = mo.image(fig_to_image(pdf_fig), width="100%")

    pdf_explanation = mo.md(
        r"""
        **Understanding the Normal Distribution Visualization:**

        - **PDF (top)**: The probability density function shows the relative likelihood of different values.
          The highest point occurs at the mean (μ).

            - **Shaded regions**: The green shaded areas represent:
                  - μ ± 1σ: Contains approximately 68.3% of the probability
                  - μ ± 2σ: Contains approximately 95.5% of the probability 
                  - μ ± 3σ: Contains approximately 99.7% of the probability (the "68-95-99.7 rule")

        - **CDF (bottom)**: The cumulative distribution function shows the probability that X is less than or equal to a given value.
              - At x = μ, the CDF equals 0.5 (50% probability)
              - At x = μ + σ, the CDF equals approximately 0.84 (84% probability)
              - At x = μ - σ, the CDF equals approximately 0.16 (16% probability)
        """
    )

    mo.vstack([pdf_image, pdf_explanation])
    return pdf_explanation, pdf_fig, pdf_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Standard Normal Distribution

        The **Standard Normal Distribution** is a special case of the normal distribution where $\mu = 0$ and $\sigma = 1$. We denote it as:

        $$Z \sim \mathcal{N}(0, 1)$$

        This distribution is particularly important because:

        1. Any normal distribution can be transformed into the standard normal
        2. Statistical tables and calculations often use the standard normal as a reference

        ### Standardizing a Normal Random Variable

        For any normal random variable $X \sim \mathcal{N}(\mu, \sigma^2)$, we can transform it to the standard normal $Z$ using:

        $$Z = \frac{X - \mu}{\sigma}$$

        Let's see the mathematical derivation:

        \begin{align*}
        W &= \frac{X -\mu}{\sigma} && \text{Subtract by $\mu$ and divide by $\sigma$} \\
          &= \frac{1}{\sigma}X - \frac{\mu}{\sigma} && \text{Use algebra to rewrite the equation}\\
          &= aX + b && \text{Linear transform where $a = \frac{1}{\sigma}$, $b = -\frac{\mu}{\sigma}$}\\
          &\sim \mathcal{N}(a\mu + b, a^2\sigma^2) && \text{The linear transform of a Normal is another Normal}\\
          &\sim \mathcal{N}\left(\frac{\mu}{\sigma} - \frac{\mu}{\sigma}, \frac{\sigma^2}{\sigma^2}\right) && \text{Substitute values for $a$ and $b$}\\
          &\sim \mathcal{N}(0, 1) && \text{The standard normal}
        \end{align*}

        This transformation is the foundation for many statistical tests and probability calculations.
        """
    )
    return


@app.cell(hide_code=True)
def _(create_standardization_plot, fig_to_image, mo):
    # Create and display visualization
    stand_fig = create_standardization_plot()

    # Display
    stand_image = mo.image(fig_to_image(stand_fig), width="100%")

    stand_explanation = mo.md(
        r"""
        **Standardizing a Normal Distribution: A Two-Step Process**

        The visualization above shows the process of transforming any normal distribution to the standard normal:

        1. **Shift the distribution** (left plot): First, we subtract the mean (μ) from X, centering the distribution at 0.

        2. **Scale the distribution** (right plot): Next, we divide by the standard deviation (σ), which adjusts the spread to match the standard normal.

        The resulting standard normal distribution Z ~ N(0,1) has a mean of 0 and a variance of 1.

        This transformation allows us to use standardized tables and calculations for any normal distribution.
        """
    )

    mo.vstack([stand_image, stand_explanation])
    return stand_explanation, stand_fig, stand_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear Transformations of Normal Variables

        One useful property of the normal distribution is that linear transformations of normal random variables remain normal.

        If $X \sim \mathcal{N}(\mu, \sigma^2)$ and $Y = aX + b$ (where $a$ and $b$ are constants), then:

        $$Y \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$$

        This means:

        - The mean is transformed by $a\mu + b$
        - The variance is transformed by $a^2\sigma^2$

        This property is extremely useful in statistics and probability calculations, as it allows us to easily determine the _distribution_ of transformed variables.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Calculating Probabilities with the Normal CDF

        Unlike many other distributions, the normal distribution does not have a closed-form expression for its CDF. However, we can use the standard normal CDF (denoted as $\Phi$) to calculate probabilities.

        For any normal random variable $X \sim \mathcal{N}(\mu, \sigma^2)$, the CDF is:

        $$F_X(x) = P(X \leq x) = \Phi\left(\frac{x - \mu}{\sigma}\right)$$

        Where $\Phi$ is the CDF of the standard normal distribution.

        ### Derivation

        \begin{align*}
        F_X(x) &= P(X \leq x) \\
        &= P\left(\frac{X - \mu}{\sigma} \leq \frac{x - \mu}{\sigma}\right) \\
        &= P\left(Z \leq \frac{x - \mu}{\sigma}\right) \\
        &= \Phi\left(\frac{x - \mu}{\sigma}\right)
        \end{align*}

        Let's look at some examples of calculating probabilities with normal distributions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Examples of Normal Distributions""")
    return


@app.cell(hide_code=True)
def _(create_probability_example, fig_to_image, mo):
    # Create visualization
    default_mu = 3
    default_sigma = 4
    default_query = 0

    prob_fig, prob_value, ex_z_score = create_probability_example(default_mu, default_sigma, default_query)

    # Display
    prob_image = mo.image(fig_to_image(prob_fig), width="100%")

    prob_explanation = mo.md(
        f"""
        **Example: Let X ~ N(3, 16), what is P(X > 0)?**

        To solve this probability question:

        1. First, we standardize the query value:
           Z = (x - μ) / σ = (0 - 3) / 4 = -0.75

        2. Then we calculate using the standard normal CDF:
           P(X > 0) = P(Z > -0.75) = 1 - P(Z ≤ -0.75) = 1 - Φ(-0.75)

        3. Because the standard normal is symmetric: 
           1 - Φ(-0.75) = Φ(0.75) = {prob_value:.3f}

        The shaded orange area in the graph represents this probability of approximately {prob_value:.3f}.
        """
    )

    mo.vstack([prob_image, prob_explanation])
    return (
        default_mu,
        default_query,
        default_sigma,
        ex_z_score,
        prob_explanation,
        prob_fig,
        prob_image,
        prob_value,
    )


@app.cell(hide_code=True)
def _(create_range_probability_example, fig_to_image, mo, stats):
    # Create visualization
    default_range_mu = 3
    default_range_sigma = 4
    default_range_lower = 2
    default_range_upper = 5

    range_fig, range_prob, range_z_lower, range_z_upper = create_range_probability_example(
        default_range_mu, default_range_sigma, default_range_lower, default_range_upper)

    # Display
    range_image = mo.image(fig_to_image(range_fig), width="100%")

    range_explanation = mo.md(
        f"""
        **Example: Let X ~ N(3, 16), what is P(2 < X < 5)?**

        To solve this range probability question:

        1. First, we standardize both bounds:
           Z_lower = (lower - μ) / σ = (2 - 3) / 4 = -0.25
           Z_upper = (upper - μ) / σ = (5 - 3) / 4 = 0.5

        2. Then we calculate using the standard normal CDF:
           P(2 < X < 5) = P(-0.25 < Z < 0.5)
           = Φ(0.5) - Φ(-0.25)
           = Φ(0.5) - (1 - Φ(0.25))
           = Φ(0.5) + Φ(0.25) - 1

        3. Computing these values:
           = {stats.norm.cdf(0.5):.3f} + {stats.norm.cdf(0.25):.3f} - 1
           = {range_prob:.3f}

        The shaded orange area in the graph represents this probability of approximately {range_prob:.3f}.
        """
    )

    mo.vstack([range_image, range_explanation])
    return (
        default_range_lower,
        default_range_mu,
        default_range_sigma,
        default_range_upper,
        range_explanation,
        range_fig,
        range_image,
        range_prob,
        range_z_lower,
        range_z_upper,
    )


@app.cell(hide_code=True)
def _(create_voltage_example_visualization, fig_to_image, mo):
    # Create vizualization
    voltage_fig, voltage_error_prob = create_voltage_example_visualization()

    # Display
    voltage_image = mo.image(fig_to_image(voltage_fig), width="100%")

    voltage_explanation = mo.md(
        r"""
        **Example: Signal Transmission with Noise**

        In this example, we're sending digital signals over a wire:

        - We send voltage 2 to represent a binary "1"
        - We send voltage -2 to represent a binary "0"

        The received signal R is the sum of the transmitted voltage (X) and random noise (Y):
        R = X + Y, where Y ~ N(0, 1)

        When decoding, we use a threshold of 0.5:

        - If R ≥ 0.5, we interpret it as "1"
        - If R < 0.5, we interpret it as "0"

        Let's calculate the probability of error when sending a "1" (voltage = 2):

        \begin{align*}
        P(\text{Error when sending "1"}) &= P(X + Y < 0.5) \\
        &= P(2 + Y < 0.5) \\
        &= P(Y < -1.5) \\
        &= \Phi(-1.5) \\
        &\approx 0.067
        \end{align*}

        Therefore, the probability of incorrectly decoding a transmitted "1" as "0" is approximately 6.7%.

        The orange shaded area in the plot represents this error probability.
        """
    )

    mo.vstack([voltage_image, voltage_explanation])
    return voltage_error_prob, voltage_explanation, voltage_fig, voltage_image


@app.cell(hide_code=True)
def emirical_rule(mo):
    mo.md(
        r"""
        ## The 68-95-99.7 Rule (Empirical Rule)

        One of the most useful properties of the normal distribution is the "[68-95-99.7 rule](https://en.wikipedia.org/wiki/68-95-99.7_rule)," which states that:

        - Approximately 68% of the data falls within 1 standard deviation of the mean
        - Approximately 95% of the data falls within 2 standard deviations of the mean
        - Approximately 99.7% of the data falls within 3 standard deviations of the mean

        Let's verify this with a calculation for the 68% rule:

        \begin{align}
        P(\mu - \sigma < X < \mu + \sigma) 
        &= P(X < \mu + \sigma) - P(X < \mu - \sigma) \\
        &= \Phi\left(\frac{(\mu + \sigma)-\mu}{\sigma}\right) - \Phi\left(\frac{(\mu - \sigma)-\mu}{\sigma}\right) \\
        &= \Phi\left(\frac{\sigma}{\sigma}\right) - \Phi\left(\frac{-\sigma}{\sigma}\right) \\
        &= \Phi(1) - \Phi(-1) \\
        &\approx 0.8413 - 0.1587 \\
        &\approx 0.6826 \approx 68.3\%
        \end{align}

        This calculation works for any normal distribution, regardless of the values of $\mu$ and $\sigma$!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The Cumulative Distribution Function (CDF) gives the probability that a random variable is less than or equal to a specific value. Use the interactive calculator below to compute CDF values for a normal distribution.""")
    return


@app.cell(hide_code=True)
def _(mo, mu_slider, sigma_slider, x_slider):
    mo.md(
        f"""
        ## Interactive Normal CDF Calculator

        Use the sliders below to explore different probability calculations:

        **Query value (x):** {x_slider} — The value at which to evaluate F(x) = P(X ≤ x)

        **Mean (μ):** {mu_slider} — The center of the distribution

        **Standard deviation (σ):** {sigma_slider} — The spread of the distribution (larger σ means more spread)
        """
    )
    return


@app.cell(hide_code=True)
def _(
    create_cdf_calculator_plot,
    fig_to_image,
    mo,
    mu_slider,
    sigma_slider,
    x_slider,
):
    # Values from widgets
    calc_x = x_slider.amount
    calc_mu = mu_slider.amount
    calc_sigma = sigma_slider.amount

    # Create visualization
    calc_fig, cdf_value = create_cdf_calculator_plot(calc_x, calc_mu, calc_sigma)

    # Standardized z-score
    calc_z_score = (calc_x - calc_mu) / calc_sigma

    # Display
    calc_image = mo.image(fig_to_image(calc_fig), width="100%")

    calc_result = mo.md(
        f"""
        ### Results:

        For a Normal distribution with parameters μ = {calc_mu:.1f} and σ = {calc_sigma:.1f}:

        - The value x = {calc_x:.1f} corresponds to a z-score of z = {calc_z_score:.3f}
        - The CDF value F({calc_x:.1f}) = P(X ≤ {calc_x:.1f}) = {cdf_value:.3f}
        - This means the probability that X is less than or equal to {calc_x:.1f} is {cdf_value*100:.1f}%

        **Computing this in Python:**
        ```python
        from scipy import stats

        # Using the one-line method
        p = stats.norm.cdf({calc_x:.1f}, {calc_mu:.1f}, {calc_sigma:.1f})

        # OR using the two-line method
        X = stats.norm({calc_mu:.1f}, {calc_sigma:.1f})
        p = X.cdf({calc_x:.1f})
        ```

        **Note:** In SciPy's `stats.norm`, the second parameter is the standard deviation (σ), not the variance (σ²).
        """
    )

    mo.vstack([calc_image, calc_result])
    return (
        calc_fig,
        calc_image,
        calc_mu,
        calc_result,
        calc_sigma,
        calc_x,
        calc_z_score,
        cdf_value,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding

        Test your knowledge with these true/false questions about normal distributions:

        /// details | For a normal random variable X ~ N(μ, σ²), the probability that X takes on exactly the value μ is highest among all possible values.

        **✅ True**

        While the PDF is indeed highest at x = μ, making this the most likely value in terms of density, remember that for continuous random variables, the probability of any exact value is zero. The statement refers to the density function being maximized at the mean.
        ///

        /// details | The probability that a normal random variable X equals any specific exact value (e.g., P(X = 3)) is always zero.

        **✅ True**

        For continuous random variables including the normal, the probability of any exact value is zero. Probabilities only make sense for ranges of values, which is why we integrate the PDF over intervals.
        ///

        /// details | If X ~ N(μ, σ²), then aX + b ~ N(aμ + b, a²σ²) for any constants a and b.

        **✅ True**

        Linear transformations of normal random variables remain normal, with the given transformation of the parameters. This is a key property that makes normal distributions particularly useful.
        ///

        /// details | If X ~ N(5, 9) and Y ~ N(3, 4) are independent, then X + Y ~ N(8, 5).

        **❌ False**

        While the mean of the sum is indeed the sum of the means (5 + 3 = 8), the variance of the sum is the sum of the variances (9 + 4 = 13), not 5. The correct distribution would be X + Y ~ N(8, 13).
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        We've taken a tour of Normal distributions; probably the most famous probability distribution you'll encounter in statistics. It's that nice bell-shaped curve that shows up everywhere from heights/ weights to memes to measurement errors & stock returns.

        The Normal distribution isn't just pretty — it's incredibly practical. With just two parameters (mean and standard deviation), you can describe complex phenomena and make powerful predictions. Plus, thanks to the Central Limit Theorem, many random processes naturally converge to this distribution, which is why it's so prevalent.

        **What we covered:**

        - The mathematical definition and key properties of Normal random variables

        - How to transform any Normal distribution to the standard Normal

        - Calculating probabilities using the CDF (no more looking up values in those tiny tables in the back of textbooks or Clark's table!)

        Whether you're analyzing data, designing experiments, or building ML models, the concepts we explored provide a solid foundation for working with this fundamental distribution.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Appendix (helper code and functions)""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    from wigglystuff import TangleSlider
    return (TangleSlider,)


@app.cell(hide_code=True)
def _(np, plt, stats):
    def create_normal_pdf_plot(mu, sigma):

        # Range for x values (show μ ± 4σ)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        pdf = stats.norm.pdf(x, mu, sigma)

        # Calculate CDF values
        cdf = stats.norm.cdf(x, mu, sigma)

        # Create plot with two subplots for (PDF and CDF)
        pdf_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # PDF plot
        ax1.plot(x, pdf, color='royalblue', linewidth=2, label='PDF')
        ax1.fill_between(x, pdf, color='royalblue', alpha=0.2)

        # Vertical line at mean
        ax1.axvline(x=mu, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Mean: μ = {mu:.1f}')

        # Stdev regions
        for i in range(1, 4):
            alpha = 0.1 if i > 1 else 0.2
            percentage = 100*stats.norm.cdf(i) - 100*stats.norm.cdf(-i)
            label = f'μ ± {i}σ: {percentage:.1f}%' if i == 1 else None
            ax1.axvspan(mu - i*sigma, mu + i*sigma, alpha=alpha, color='green', 
                       label=label)

        # Annotations
        ax1.annotate(f'μ = {mu:.1f}', xy=(mu, max(pdf)*0.15), xytext=(mu+0.5*sigma, max(pdf)*0.4),
                    arrowprops=dict(facecolor='black', width=1, shrink=0.05))

        ax1.annotate(f'σ = {sigma:.1f}', 
                    xy=(mu+sigma, stats.norm.pdf(mu+sigma, mu, sigma)), 
                    xytext=(mu+1.5*sigma, stats.norm.pdf(mu+sigma, mu, sigma)*1.5),
                    arrowprops=dict(facecolor='black', width=1, shrink=0.05))

        # some styling
        ax1.set_title(f'Normal Distribution PDF: N({mu:.1f}, {sigma:.1f}²)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability Density: f(x)')
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.3)

        # CDF plot
        ax2.plot(x, cdf, color='darkorange', linewidth=2, label='CDF')

        # key CDF values mark
        key_points = [
            (mu-sigma, stats.norm.cdf(mu-sigma, mu, sigma), "16%"),
            (mu, 0.5, "50%"),
            (mu+sigma, stats.norm.cdf(mu+sigma, mu, sigma), "84%")
        ]

        for point, value, label in key_points:
            ax2.plot(point, value, 'ro')
            ax2.annotate(f'{label}', 
                        xy=(point, value),
                        xytext=(point+0.2*sigma, value-0.1),
                        arrowprops=dict(facecolor='black', width=1, shrink=0.05))

        # CDF styling
        ax2.set_title(f'Normal Distribution CDF: N({mu:.1f}, {sigma:.1f}²)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Cumulative Probability: F(x)')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        return pdf_fig
    return (create_normal_pdf_plot,)


@app.cell(hide_code=True)
def _(base64, io):
    from matplotlib.figure import Figure

    # convert matplotlib figures to images (helper code)
    def fig_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    return Figure, fig_to_image


@app.cell(hide_code=True)
def _():
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import io
    import base64
    return base64, io, np, plt, stats


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    mean_slider = mo.ui.anywidget(TangleSlider(
        amount=0, 
        min_value=-5, 
        max_value=5, 
        step=0.1,
        digits=1
    ))

    std_slider = mo.ui.anywidget(TangleSlider(
        amount=1, 
        min_value=0.1, 
        max_value=3, 
        step=0.1,
        digits=1
    ))
    return mean_slider, std_slider


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    x_slider = mo.ui.anywidget(TangleSlider(
        amount=0,
        min_value=-5,
        max_value=5,
        step=0.1,
        digits=1
    ))

    mu_slider = mo.ui.anywidget(TangleSlider(
        amount=0,
        min_value=-5,
        max_value=5,
        step=0.1,
        digits=1
    ))

    sigma_slider = mo.ui.anywidget(TangleSlider(
        amount=1,
        min_value=0.1,
        max_value=3,
        step=0.1,
        digits=1
    ))
    return mu_slider, sigma_slider, x_slider


@app.cell(hide_code=True)
def _(np, plt, stats):
    def create_distribution_comparison(mu=5, sigma=6):

        # Create figure and axis
        comparison_fig, ax = plt.subplots(figsize=(10, 6))

        # X range for plotting
        x = np.linspace(-10, 20, 1000)

        # Standard normal
        std_normal = stats.norm.pdf(x, 0, 1)

        # Our example normal
        example_normal = stats.norm.pdf(x, mu, sigma)

        # Plot both distributions
        ax.plot(x, std_normal, 'darkviolet', linewidth=2, label='Standard Normal')
        ax.plot(x, example_normal, 'blue', linewidth=2, label=f'X ~ N({mu}, {sigma}²)')

        # format the plot
        ax.set_xlim(-10, 20)
        ax.set_ylim(0, 0.45)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Decorative text box for parameters
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        textstr = '\n'.join((
            r'Normal (aka Gaussian) Random Variable',
            r'',
            f'Parameter $\mu$: {mu}',
            f'Parameter $\sigma$: {sigma}'
        ))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        return comparison_fig
    return (create_distribution_comparison,)


@app.cell(hide_code=True)
def _(np, plt, stats):
    def create_voltage_example_visualization():

        # Create data for plotting
        x = np.linspace(-4, 4, 1000)

        # Signal without noise (X = 2)
        signal_value = 2

        # Noise distribution (Y ~ N(0, 1))
        noise_pdf = stats.norm.pdf(x, 0, 1)

        # Signal + Noise distribution (R = X + Y ~ N(2, 1))
        received_pdf = stats.norm.pdf(x, signal_value, 1)

        # Create figure
        voltage_fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the noise distribution
        ax.plot(x, noise_pdf, 'blue', linewidth=1.5, alpha=0.6, 
               label='Noise: Y ~ N(0, 1)')

        # received signal distribution
        ax.plot(x, received_pdf, 'red', linewidth=2, 
               label=f'Received: R ~ N({signal_value}, 1)')

        # vertical line at the decision boundary (0.5)
        threshold = 0.5
        ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                  label=f'Decision threshold: {threshold}')

        # Shade the error region
        mask = x < threshold
        error_prob = stats.norm.cdf(threshold, signal_value, 1)
        ax.fill_between(x[mask], received_pdf[mask], color='darkorange', alpha=0.5,
                       label=f'Error probability: {error_prob:.3f}')

        # Styling
        ax.set_title('Voltage Transmission Example: Probability of Error')
        ax.set_xlabel('Voltage')
        ax.set_ylabel('Probability Density')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)

        # Add explanatory annotations
        ax.text(1.5, 0.1, 'When sending "1" (voltage=2),\nthis area represents\nthe error probability', 
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

        plt.tight_layout()
        plt.gca()
        return voltage_fig, error_prob
    return (create_voltage_example_visualization,)


@app.cell(hide_code=True)
def _(np, plt, stats):
    def create_cdf_calculator_plot(calc_x, calc_mu, calc_sigma):

        # Data range for plotting
        x_range = np.linspace(calc_mu - 4*calc_sigma, calc_mu + 4*calc_sigma, 1000)
        pdf = stats.norm.pdf(x_range, calc_mu, calc_sigma)
        cdf = stats.norm.cdf(x_range, calc_mu, calc_sigma)

        # Calculate the CDF at x
        cdf_at_x = stats.norm.cdf(calc_x, calc_mu, calc_sigma)

        # Create figure with two subplots
        calc_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot PDF on top subplot
        ax1.plot(x_range, pdf, color='royalblue', linewidth=2, label='PDF')

        # area shade for P(X ≤ x)
        mask = x_range <= calc_x
        ax1.fill_between(x_range[mask], pdf[mask], color='darkorange', alpha=0.6)

        # Vertical line at x
        ax1.axvline(x=calc_x, color='red', linestyle='--', linewidth=1.5)

        # PDF labels and styling
        ax1.set_title(f'Normal PDF with Area P(X ≤ {calc_x:.1f}) Highlighted')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability Density')
        ax1.annotate(f'x = {calc_x:.1f}', xy=(calc_x, 0), xytext=(calc_x, -0.01),
                    horizontalalignment='center', color='red')
        ax1.grid(alpha=0.3)

        # CDF on bottom subplot
        ax2.plot(x_range, cdf, color='green', linewidth=2, label='CDF')

        # Mark the point (x, CDF(x))
        ax2.plot(calc_x, cdf_at_x, 'ro', markersize=8)

        # CDF labels and styling
        ax2.set_title(f'Normal CDF: F({calc_x:.1f}) = {cdf_at_x:.3f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Cumulative Probability')
        ax2.annotate(f'F({calc_x:.1f}) = {cdf_at_x:.3f}', 
                     xy=(calc_x, cdf_at_x), 
                     xytext=(calc_x + 0.5*calc_sigma, cdf_at_x - 0.1),
                     arrowprops=dict(facecolor='black', width=1, shrink=0.05),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.gca()
        return calc_fig, cdf_at_x
    return (create_cdf_calculator_plot,)


@app.cell(hide_code=True)
def _(np, plt, stats):
    def create_standardization_plot():

        x = np.linspace(-6, 6, 1000)

        # Original distribution N(2, 1.5²)
        mu_original, sigma_original = 2, 1.5
        pdf_original = stats.norm.pdf(x, mu_original, sigma_original)

        # shifted distribution N(0, 1.5²)
        mu_shifted, sigma_shifted = 0, 1.5
        pdf_shifted = stats.norm.pdf(x, mu_shifted, sigma_shifted)

        # Standard normal N(0, 1)
        mu_standard, sigma_standard = 0, 1
        pdf_standard = stats.norm.pdf(x, mu_standard, sigma_standard)

        # Create visualization
        stand_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot on  left: Original and shifted distributions
        ax1.plot(x, pdf_original, 'royalblue', linewidth=2, 
                label=f'Original: N({mu_original}, {sigma_original}²)')
        ax1.plot(x, pdf_shifted, 'darkorange', linewidth=2, 
                label=f'Shifted: N({mu_shifted}, {sigma_shifted}²)')

        # Add arrow to show the shift
        shift_x1, shift_y1 = mu_original, stats.norm.pdf(mu_original, mu_original, sigma_original)*0.6
        shift_x2, shift_y2 = mu_shifted, stats.norm.pdf(mu_shifted, mu_shifted, sigma_shifted)*0.6
        ax1.annotate('', xy=(shift_x2, shift_y2), xytext=(shift_x1, shift_y1),
                    arrowprops=dict(facecolor='black', width=1.5, shrink=0.05))
        ax1.text(0.8, 0.28, 'Subtract μ', transform=ax1.transAxes)

        # Plot on right: Shifted and standard normal
        ax2.plot(x, pdf_shifted, 'darkorange', linewidth=2, 
                label=f'Shifted: N({mu_shifted}, {sigma_shifted}²)')
        ax2.plot(x, pdf_standard, 'green', linewidth=2, 
                label=f'Standard: N({mu_standard}, {sigma_standard}²)')

        # Add arrow to show the scaling
        scale_x1, scale_y1 = 2*sigma_shifted, stats.norm.pdf(2*sigma_shifted, mu_shifted, sigma_shifted)*0.8
        scale_x2, scale_y2 = 2*sigma_standard, stats.norm.pdf(2*sigma_standard, mu_standard, sigma_standard)*0.8
        ax2.annotate('', xy=(scale_x2, scale_y2), xytext=(scale_x1, scale_y1),
                    arrowprops=dict(facecolor='black', width=1.5, shrink=0.05))
        ax2.text(0.75, 0.5, 'Divide by σ', transform=ax2.transAxes)

        # some styling
        for ax in (ax1, ax2):
            ax.set_xlabel('x')
            ax.set_ylabel('Probability Density')
            ax.grid(alpha=0.3)
            ax.legend()

        ax1.set_title('Step 1: Shift the Distribution')
        ax2.set_title('Step 2: Scale the Distribution')

        plt.tight_layout()
        plt.gca()
        return stand_fig
    return (create_standardization_plot,)


@app.cell(hide_code=True)
def _(np, plt, stats):
    def create_probability_example(example_mu=3, example_sigma=4, example_query=0):

        # Create data range
        x = np.linspace(example_mu - 4*example_sigma, example_mu + 4*example_sigma, 1000)
        pdf = stats.norm.pdf(x, example_mu, example_sigma)

        # probability calc
        prob_value = 1 - stats.norm.cdf(example_query, example_mu, example_sigma)
        ex_z_score = (example_query - example_mu) / example_sigma

        # Create visualization
        prob_fig, ax = plt.subplots(figsize=(10, 6))

        # Plot PDF
        ax.plot(x, pdf, 'royalblue', linewidth=2)

        # area shading representing the probability
        mask = x >= example_query
        ax.fill_between(x[mask], pdf[mask], color='darkorange', alpha=0.6)

        # Add vertical line at query point
        ax.axvline(x=example_query, color='red', linestyle='--', linewidth=1.5)

        # Annotations
        ax.annotate(f'x = {example_query}', xy=(example_query, 0), xytext=(example_query, -0.005),
                   horizontalalignment='center')

        ax.annotate(f'P(X > {example_query}) = {prob_value:.3f}', 
                    xy=(example_query + example_sigma, 0.015), 
                    xytext=(example_query + 1.5*example_sigma, 0.02),
                    arrowprops=dict(facecolor='black', width=1, shrink=0.05),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

        # Standard normal calculation annotation
        ax.annotate(f'= P(Z > {ex_z_score:.3f}) = {prob_value:.3f}', 
                    xy=(example_query - example_sigma, 0.01), 
                    xytext=(example_query - 2*example_sigma, 0.015),
                    arrowprops=dict(facecolor='black', width=1, shrink=0.05),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

        # some styling
        ax.set_title(f'Example: P(X > {example_query}) where X ~ N({example_mu}, {example_sigma}²)')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.gca()
        return prob_fig, prob_value, ex_z_score
    return (create_probability_example,)


@app.cell(hide_code=True)
def _(np, plt, stats):
    def create_range_probability_example(range_mu=3, range_sigma=4, range_lower=2, range_upper=5):

        x = np.linspace(range_mu - 4*range_sigma, range_mu + 4*range_sigma, 1000)
        pdf = stats.norm.pdf(x, range_mu, range_sigma)

        # probability
        range_prob = stats.norm.cdf(range_upper, range_mu, range_sigma) - stats.norm.cdf(range_lower, range_mu, range_sigma)
        range_z_lower = (range_lower - range_mu) / range_sigma
        range_z_upper = (range_upper - range_mu) / range_sigma

        # Create visualization
        range_fig, ax = plt.subplots(figsize=(10, 6))

        # Plot PDF
        ax.plot(x, pdf, 'royalblue', linewidth=2)

        # Shade the area representing the probability
        mask = (x >= range_lower) & (x <= range_upper)
        ax.fill_between(x[mask], pdf[mask], color='darkorange', alpha=0.6)

        # Add vertical lines at query points
        ax.axvline(x=range_lower, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(x=range_upper, color='red', linestyle='--', linewidth=1.5)

        # Annotations
        ax.annotate(f'x = {range_lower}', xy=(range_lower, 0), xytext=(range_lower, -0.005),
                   horizontalalignment='center')
        ax.annotate(f'x = {range_upper}', xy=(range_upper, 0), xytext=(range_upper, -0.005),
                   horizontalalignment='center')

        ax.annotate(f'P({range_lower} < X < {range_upper}) = {range_prob:.3f}', 
                    xy=((range_lower + range_upper)/2, max(pdf[mask])/2), 
                    xytext=((range_lower + range_upper)/2, max(pdf[mask])*1.5),
                    arrowprops=dict(facecolor='black', width=1, shrink=0.05),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
                    horizontalalignment='center')

        # Standard normal calculation annotation
        ax.annotate(f'= P({range_z_lower:.3f} < Z < {range_z_upper:.3f}) = {range_prob:.3f}', 
                    xy=((range_lower + range_upper)/2, max(pdf[mask])/3), 
                    xytext=(range_mu - 2*range_sigma, max(pdf[mask])/1.5),
                    arrowprops=dict(facecolor='black', width=1, shrink=0.05),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

        ax.set_title(f'Example: P({range_lower} < X < {range_upper}) where X ~ N({range_mu}, {range_sigma}²)')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.gca()
        return range_fig, range_prob, range_z_lower, range_z_upper
    return (create_range_probability_example,)


if __name__ == "__main__":
    app.run()

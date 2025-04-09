# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "altair==5.5.0",
#     "matplotlib==3.10.1",
#     "numpy==2.2.4",
#     "scipy==1.15.2",
#     "sympy==1.13.3",
#     "wigglystuff==0.1.10",
#     "polars==1.26.0",
# ]
# ///

import marimo

__generated_with = "0.12.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Continuous Distributions

        _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/continuous/), by Stanford professor Chris Piech._

        Continuous distributions are what we need when dealing with random variables that can take any value in a range, rather than just discrete values. 

        The key difference here is that we work with probability density functions (PDFs) instead of probability mass functions (PMFs). It took me a while to really get this - the PDF at a point isn't actually a probability, but rather a density.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## From Discrete to Continuous

        Making the jump from discrete to continuous random variables requires a fundamental shift in thinking. Let me walk you through a thought experiment:

        > You're rushing to catch a bus. You know you'll arrive at 2:15pm, but the bus arrival time is uncertain. If you model the bus arrival time (in minutes past 2pm) as a random variable $T$, how would you calculate the probability of waiting more than five minutes: $P(15 < T < 20)$?

        This highlights a crucial difference from discrete distributions. With discrete distributions, we calculated probabilities for exact values, but this approach breaks down with continuous values like time.

        Consider these questions:
        - What's the probability the bus arrives at exactly 2:17pm and 12.12333911102389234 seconds?
        - What's the probability a newborn weighs exactly 3.523112342234 kilograms?

        These questions have no meaningful answers because continuous measurements can have infinite precision. In the continuous world, the probability of a random variable taking any specific exact value is actually zero!

        Let's visualize this transition from discrete to continuous:
        """
    )
    return


@app.cell(hide_code=True)
def _(fig_to_image, mo, np, plt):
    def create_discretization_plot():
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # values from 0 to 30 minutes
        x = np.linspace(0, 30, 1000)

        # Triangular distribution peaked at 15 minutes)
        y = np.where(x <= 15, x/15, (30-x)/15)
        y = y / np.trapezoid(y, x)  # Normalize

        # 5-minute chunks (first plot)
        bins = np.arange(0, 31, 5)
        hist, _ = np.histogram(x, bins=bins, weights=y)
        width = bins[1] - bins[0]
        axs[0].bar(bins[:-1], hist * width, width=width, alpha=0.7, 
                   color='royalblue', edgecolor='black')
        axs[0].set_xlim(0, 30)
        axs[0].set_title('5-Minute Intervals')
        axs[0].set_xlabel('Minutes past 2pm')
        axs[0].set_ylabel('Probability')

        # 15-20 minute range more prominent
        axs[0].bar([15], hist[3] * width, width=width, alpha=0.7, 
                   color='darkorange', edgecolor='black')

        # 2.5-minute chunks (second plot)
        bins = np.arange(0, 31, 2.5)
        hist, _ = np.histogram(x, bins=bins, weights=y)
        width = bins[1] - bins[0]
        axs[1].bar(bins[:-1], hist * width, width=width, alpha=0.7, 
                   color='royalblue', edgecolor='black')
        axs[1].set_xlim(0, 30)
        axs[1].set_title('2.5-Minute Intervals')
        axs[1].set_xlabel('Minutes past 2pm')

        # Make 15-20 minute range more prominent
        highlight_indices = [6, 7]
        for idx in highlight_indices:
            axs[1].bar([bins[idx]], hist[idx] * width, width=width, alpha=0.7, 
                       color='darkorange', edgecolor='black')

        # Continuous distribution (third plot)
        axs[2].plot(x, y, 'royalblue', linewidth=2)
        axs[2].set_xlim(0, 30)
        axs[2].set_title('Continuous Distribution')
        axs[2].set_xlabel('Minutes past 2pm')
        axs[2].set_ylabel('Probability Density')

        # Highlight the AUC between 15 and 20
        mask = (x >= 15) & (x <= 20)
        axs[2].fill_between(x[mask], y[mask], color='darkorange', alpha=0.7)

        # Mark 15-20 minute interval
        for ax in axs:
            ax.axvline(x=15, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=20, color='red', linestyle='--', alpha=0.5)
            ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.gca()
        return fig

    # Plot creation & conversion
    _fig = create_discretization_plot()
    _img = mo.image(fig_to_image(_fig), width="100%")

    _explanation = mo.md(
        r"""
        The figure above illustrates our transition from discrete to continuous thinking:

        - **Left**: Time divided into 5-minute chunks, where the probability of the bus arriving between 15-20 minutes (highlighted in orange) is a single value.
        - **Center**: Time divided into finer 2.5-minute chunks, where the 15-20 minute range consists of two chunks.
        - **Right**: In the limit, we get a continuous probability density function where the probability is the area under the curve between 15 and 20 minutes.

        As we make our chunks smaller and smaller, we eventually arrive at a smooth function that gives us the probability density at each point.
        """
    )

    mo.vstack([_img, _explanation])
    return (create_discretization_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Probability Density Functions

        While discrete random variables use Probability Mass Functions (PMFs), continuous random variables require a different approach — Probability Density Functions (PDFs).

        A PDF defines the relative likelihood of a continuous random variable taking particular values. We typically denote this with $f$ and write it as:

        $$f(X=x) \quad \text{or simply} \quad f(x)$$

        Where the lowercase $x$ represents a specific value our random variable $X$ might take.

        ### Key Properties of PDFs

        For a PDF $f(x)$ to be valid, it must satisfy these properties:

        1. The probability that $X$ falls within interval $[a, b]$ is:

           $$P(a \leq X \leq b) = \int_a^b f(x) \, dx$$

        2. Non-negativity — the PDF can't be negative:

           $$f(x) \geq 0 \text{ for all } x$$

        3. Total probability equals 1:

           $$\int_{-\infty}^{\infty} f(x) \, dx = 1$$

        4. The probability of any exact value is zero:

           $$P(X = a) = \int_a^a f(x) \, dx = 0$$

        This last property reveals a fundamental difference from discrete distributions — with continuous random variables, probabilities only make sense for ranges, not specific points.

        ### Important Distinction: Density ≠ Probability

        One common mistake is interpreting $f(x)$ as a probability. It's actually a **density** — representing probability per unit of $x$. This is why $f(x)$ values can exceed 1, provided the total area under the curve equals 1.

        The true meaning of $f(x)$ emerges only when:
        1. We integrate over a range to obtain an actual probability, or
        2. We compare densities at different points to understand relative likelihoods.
        """
    )
    return


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    # Create sliders for a and b
    a_slider = mo.ui.anywidget(TangleSlider(
        amount=1, 
        min_value=0, 
        max_value=5, 
        step=0.1,
        digits=1
    ))

    b_slider = mo.ui.anywidget(TangleSlider(
        amount=3, 
        min_value=0, 
        max_value=5, 
        step=0.1,
        digits=1
    ))

    # Distribution selector
    distribution_radio = mo.ui.radio(
        options=["uniform", "triangular", "exponential"],
        value="uniform",
        label="Distribution Type"
    )

    # Controls layout
    _controls = mo.vstack([
        mo.md("### Visualizing Probability as Area Under the PDF Curve"),
        mo.md("Adjust sliders to change the interval $[a, b]$ and see how the probability changes:"),
        mo.hstack([
            mo.md("Lower bound (a):"),
            a_slider,
            mo.md("Upper bound (b):"),
            b_slider
        ], justify="start"),
        distribution_radio
    ])
    _controls
    return a_slider, b_slider, distribution_radio


@app.cell(hide_code=True)
def _(
    a_slider,
    b_slider,
    create_pdf_visualization,
    distribution_radio,
    fig_to_image,
    mo,
):
    a = a_slider.amount
    b = b_slider.amount
    distribution = distribution_radio.value

    # Ensure a < b
    if a > b:
        a, b = b, a

    # visualization
    _fig, _probability = create_pdf_visualization(a, b, distribution)

    # Display visualization
    _img = mo.image(fig_to_image(_fig), width="100%")

    # Add appropriate explanation
    if distribution == "uniform":
        _explanation = mo.md(
            f"""
            In the **uniform distribution**, all values between 0 and 5 are equally likely. 
            The probability density is constant at 0.2 (which is 1/5, ensuring the total area is 1).
            For a uniform distribution, the probability that $X$ is in the interval $[{a:.1f}, {b:.1f}]$ 
            is simply proportional to the width of the interval: $P({a:.1f} \leq X \leq {b:.1f}) = {_probability:.4f}$
            Note that while the PDF has a constant value of 0.2, this is not a probability but a density!
            """
        )
    elif distribution == "triangular":
        _explanation = mo.md(
            f"""
            In this **triangular distribution**, the probability density increases linearly from 0 to 2.5, 
            then decreases linearly from 2.5 to 5.
            The distribution's peak is at x = 2.5, where the value is highest.
            The orange shaded area representing $P({a:.1f} \leq X \leq {b:.1f}) = {_probability:.4f}$ 
            is calculated by integrating the PDF over the interval.
            """
        )
    else:
        _explanation = mo.md(
            f"""
            The **exponential distribution** (with λ = 0.5) models the time between events in a Poisson process.
            Unlike the uniform and triangular distributions, the exponential distribution has infinite support 
            (extends from 0 to infinity). The probability density decreases exponentially as x increases.
            The orange shaded area representing $P({a:.1f} \leq X \leq {b:.1f}) = {_probability:.4f}$ 
            is calculated by integrating $f(x) = 0.5e^{{-0.5x}}$ over the interval.
            """
        )
    mo.vstack([_img, _explanation])
    return a, b, distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Cumulative Distribution Function

        Since working with PDFs requires solving integrals to find probabilities, we often use the **Cumulative Distribution Function (CDF)** as a more convenient tool.

        The CDF $F(x)$ for a continuous random variable $X$ is defined as:

        $$F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t)\,dt$$

        where $f(t)$ is the PDF of $X$.

        ### Properties of CDFs

        A CDF $F(x)$ has these key properties:

        1. $F(x)$ is always non-decreasing: if $a < b$, then $F(a) \leq F(b)$
        2. $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$
        3. $F(x)$ is right-continuous: $\lim_{h \to 0^+} F(x+h) = F(x)$

        ### Using the CDF to Calculate Probabilities

        The CDF is extremely useful because it allows us to calculate various probabilities without having to perform integrals each time:

        | Probability Query | Solution | Explanation |
        |-------------------|----------|-------------|
        | $P(X < a)$ | $F(a)$ | Definition of the CDF |
        | $P(X \leq a)$ | $F(a)$ | For continuous distributions, $P(X = a) = 0$ |
        | $P(X > a)$ | $1 - F(a)$ | Since $P(X \leq a) + P(X > a) = 1$ |
        | $P(a < X < b)$ | $F(b) - F(a)$ | Since $F(a) + P(a < X < b) = F(b)$ |
        | $P(a \leq X \leq b)$ | $F(b) - F(a)$ | Since $P(X = a) = P(X = b) = 0$ |

        For discrete random variables, the CDF is also defined but it's less commonly used:

        $$F_X(a) = \sum_{i \leq a} P(X = i)$$

        The CDF for discrete distributions is a step function, increasing at each point in the support of the random variable.
        """
    )
    return


@app.cell(hide_code=True)
def _(fig_to_image, mo, np, plt):
    def create_pdf_cdf_comparison():
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))

        # x-values
        x = np.linspace(-1, 6, 1000)

        # 1. Uniform Distribution
        # PDF
        pdf_uniform = np.where((x >= 0) & (x <= 5), 0.2, 0)
        axs[0, 0].plot(x, pdf_uniform, 'b-', linewidth=2)
        axs[0, 0].set_title('Uniform PDF')
        axs[0, 0].set_ylabel('Density')
        axs[0, 0].grid(alpha=0.3)

        # CDF
        cdf_uniform = np.zeros_like(x)
        for i, val in enumerate(x):
            if val < 0:
                cdf_uniform[i] = 0
            elif val > 5:
                cdf_uniform[i] = 1
            else:
                cdf_uniform[i] = val / 5

        axs[0, 1].plot(x, cdf_uniform, 'r-', linewidth=2)
        axs[0, 1].set_title('Uniform CDF')
        axs[0, 1].set_ylabel('Probability')
        axs[0, 1].grid(alpha=0.3)

        # 2. Triangular Distribution
        # PDF
        pdf_triangular = np.where(x <= 2.5, x/6.25, (5-x)/6.25)
        pdf_triangular = np.where((x < 0) | (x > 5), 0, pdf_triangular)

        axs[1, 0].plot(x, pdf_triangular, 'b-', linewidth=2)
        axs[1, 0].set_title('Triangular PDF')
        axs[1, 0].set_ylabel('Density')
        axs[1, 0].grid(alpha=0.3)

        # CDF
        cdf_triangular = np.zeros_like(x)
        for i, val in enumerate(x):
            if val <= 0:
                cdf_triangular[i] = 0
            elif val >= 5:
                cdf_triangular[i] = 1
            else:
                # For x ≤ 2.5: CDF = x²/(2 *6 .25)
                # For x > 2.5: CDF = 1 - (5 - x)²/(2 * 6.25)
                if val <= 2.5:
                    cdf_triangular[i] = (val**2) / (2 * 6.25)
                else:
                    cdf_triangular[i] = 1 - ((5 - val)**2) / (2 * 6.25)

        axs[1, 1].plot(x, cdf_triangular, 'r-', linewidth=2)
        axs[1, 1].set_title('Triangular CDF')
        axs[1, 1].set_ylabel('Probability')
        axs[1, 1].grid(alpha=0.3)

        # 3. Exponential Distribution
        # PDF
        lambda_param = 0.5
        pdf_exponential = np.where(x >= 0, lambda_param * np.exp(-lambda_param * x), 0)

        axs[2, 0].plot(x, pdf_exponential, 'b-', linewidth=2)
        axs[2, 0].set_title('Exponential PDF (λ=0.5)')
        axs[2, 0].set_xlabel('x')
        axs[2, 0].set_ylabel('Density')
        axs[2, 0].grid(alpha=0.3)

        # CDF
        cdf_exponential = np.where(x < 0, 0, 1 - np.exp(-lambda_param * x))

        axs[2, 1].plot(x, cdf_exponential, 'r-', linewidth=2)
        axs[2, 1].set_title('Exponential CDF (λ=0.5)')
        axs[2, 1].set_xlabel('x')
        axs[2, 1].set_ylabel('Probability')
        axs[2, 1].grid(alpha=0.3)

        # Common x-limits
        for ax in axs.flatten():
            ax.set_xlim(-0.5, 5.5)
            if ax in axs[:, 0]:  # PDF plots
                ax.set_ylim(-0.05, max(0.5, max(pdf_triangular)*1.1))
            else:  # CDF plots
                ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.gca()
        return fig

    # Create visualization
    _fig = create_pdf_cdf_comparison()
    _img = mo.image(fig_to_image(_fig), width="100%")

    _explanation = mo.md(
        r"""
        The figure above compares the Probability Density Functions (PDFs) on the left with their corresponding Cumulative Distribution Functions (CDFs) on the right for three common distributions:

        1. **Uniform Distribution**: 

           - PDF: Constant value (0.2) across the support range [0, 5]
           - CDF: Linear increase from 0 to 1 across the support range

        2. **Triangular Distribution**:

           - PDF: Linearly increases then decreases, forming a triangle shape
           - CDF: Increases quadratically up to the peak, then approaches 1 quadratically

        3. **Exponential Distribution**:

           - PDF: Starts at λ=0.5 and decreases exponentially
           - CDF: Starts at 0 and approaches 1 exponentially (never quite reaching 1)

        /// NOTE
        The common properties of all CDFs:

        - They are non-decreasing functions
        - They start at 0 (for x = -∞) and approach or reach 1 (for x = ∞)
        - The slope of the CDF at any point equals the PDF value at that point
        """
    )

    mo.vstack([_img, _explanation])
    return (create_pdf_cdf_comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Solving for Constants in PDFs

        Many PDFs contain a constant that needs to be determined to ensure the total probability equals 1. Let's work through an example to understand how to solve for these constants.

        ### Example: Finding the Constant $C$

        Let $X$ be a continuous random variable with PDF:

        $$f(x) = \begin{cases} 
        C(4x - 2x^2) & \text{when } 0 < x < 2 \\ 
        0 & \text{otherwise} 
        \end{cases}$$

        In this function, $C$ is a constant we need to determine. Since we know the PDF must integrate to 1:

        \begin{align}
        &\int_0^2 C(4x - 2x^2) \, dx = 1 \\
        &C\left(2x^2 - \frac{2x^3}{3}\right)\bigg|_0^2 = 1 \\
        &C\left[\left(8 - \frac{16}{3}\right) - 0 \right] = 1 \\
        &C\left(\frac{24 - 16}{3}\right) = 1 \\
        &C\left(\frac{8}{3}\right) = 1 \\
        &C = \frac{3}{8}
        \end{align}

        Now that we know $C = \frac{3}{8}$, we can compute probabilities. For example, what is $P(X > 1)$?

        \begin{align}
        P(X > 1) 
            &= \int_1^{\infty}f(x) \, dx \\
            &= \int_1^2 \frac{3}{8}(4x - 2x^2) \, dx \\
            &= \frac{3}{8}\left(2x^2 - \frac{2x^3}{3}\right)\bigg|_1^2 \\
            &= \frac{3}{8}\left[\left(8 - \frac{16}{3}\right) - \left(2 - \frac{2}{3}\right)\right] \\
            &= \frac{3}{8}\left[\left(8 - \frac{16}{3}\right) - \left(\frac{6 - 2}{3}\right)\right] \\
            &= \frac{3}{8}\left[\left(\frac{24 - 16}{3}\right) - \left(\frac{4}{3}\right)\right] \\
            &= \frac{3}{8}\left[\left(\frac{8}{3}\right) - \left(\frac{4}{3}\right)\right] \\
            &= \frac{3}{8} \cdot \frac{4}{3} \\
            &= \frac{1}{2}
        \end{align}

        Let's visualize this distribution and verify our results:
        """
    )
    return


@app.cell(hide_code=True)
def _(
    create_example_pdf_visualization,
    fig_to_image,
    mo,
    symbolic_calculation,
):
    # Create visualization
    _fig = create_example_pdf_visualization()
    _img = mo.image(fig_to_image(_fig), width="100%")

    # Symbolic calculation
    _sympy_verification = mo.md(symbolic_calculation())

    _explanation = mo.md(
        r"""
        The figure above shows:

        1. **Left**: The PDF $f(x) = \frac{3}{8}(4x - 2x^2)$ for $0 < x < 2$, with the area representing P(X > 1) shaded in orange.
        2. **Right**: The corresponding CDF, showing F(1) = 0.5 and thus P(X > 1) = 1 - F(1) = 0.5.

        Notice how we:

        1. First determined the constant C = 3/8 by ensuring the total area under the PDF equals 1
        2. Used this value to calculate specific probabilities like P(X > 1)
        3. Verified our results both graphically and symbolically
        """
    )
    mo.vstack([_img, _sympy_verification, _explanation])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Expectation and Variance of Continuous Random Variables

        Just as with discrete random variables, we can calculate the expectation and variance of continuous random variables. The main difference is that we use integrals instead of sums.

        ### Expectation (Mean)

        For a continuous random variable $X$ with PDF $f(x)$, the expectation is:

        $$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

        More generally, for any function $g(X)$:

        $$E[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f(x) \, dx$$

        ### Variance

        The variance is defined the same way as for discrete random variables:

        $$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

        where $\mu = E[X]$ is the mean of $X$.

        To calculate $E[X^2]$, we use:

        $$E[X^2] = \int_{-\infty}^{\infty} x^2 \cdot f(x) \, dx$$

        ### Properties

        The following properties hold for both continuous and discrete random variables:

        1. $E[aX + b] = aE[X] + b$ for constants $a$ and $b$
        2. $\text{Var}(aX + b) = a^2 \text{Var}(X)$ for constants $a$ and $b$

        Let's calculate the expectation and variance for our example PDF:
        """
    )
    return


@app.cell(hide_code=True)
def _(fig_to_image, mo, np, plt, sympy):
    # Symbolic calculation of expectation and variance
    def symbolic_stats_calc():
        x = sympy.symbols('x')
        C = sympy.Rational(3, 8)

        # Define the PDF
        pdf_expr = C * (4*x - 2*x**2)

        # Calculate expectation
        E_X = sympy.integrate(x * pdf_expr, (x, 0, 2))

        # Calculate E[X²]
        E_X2 = sympy.integrate(x**2 * pdf_expr, (x, 0, 2))

        # Calculate variance
        Var_X = E_X2 - E_X**2

        # Calculate standard deviation
        Std_X = sympy.sqrt(Var_X)

        return E_X, E_X2, Var_X, Std_X

    # Get symbolic results
    E_X, E_X2, Var_X, Std_X = symbolic_stats_calc()

    # Numerical values for plotting
    E_X_val = float(E_X)
    Var_X_val = float(Var_X)
    Std_X_val = float(Std_X)

    def create_expectation_variance_vis():
        """Create visualization showing mean and variance for the example PDF."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # x-values
        x = np.linspace(-0.5, 2.5, 1000)

        # PDF function
        C = 3/8
        pdf = np.where((x > 0) & (x < 2), C * (4*x - 2*x**2), 0)

        # Plot the PDF
        ax.plot(x, pdf, 'b-', linewidth=2, label='PDF')
        ax.fill_between(x, pdf, where=(x > 0) & (x < 2), alpha=0.3, color='blue')

        # Mark the mean
        ax.axvline(x=E_X_val, color='r', linestyle='--', linewidth=2, 
                   label=f'Mean (E[X] = {E_X_val:.3f})')

        # Mark the standard deviation range
        ax.axvspan(E_X_val - Std_X_val, E_X_val + Std_X_val, alpha=0.2, color='green',
                  label=f'±1 Std Dev ({Std_X_val:.3f})')

        # Add labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.set_title('PDF with Mean and Variance')
        ax.legend()
        ax.grid(alpha=0.3)

        # Set x-limits
        ax.set_xlim(-0.25, 2.25)

        plt.tight_layout()
        return fig

    # Create the visualization
    _fig = create_expectation_variance_vis()
    _img = mo.image(fig_to_image(_fig), width="100%")

    # Detailed calculations for our example
    _calculations = mo.md(
        f"""
        ### Computing Expectation and Variance

        > _Note:_ The following mathematical derivation is included as reference material. The credit for this approach belongs to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/continuous/) by Chris Piech.

        Let's work through the calculations for our PDF:

        $$f(x) = \\begin{{cases}} 
        \\frac{{3}}{{8}}(4x - 2x^2) & \\text{{when }} 0 < x < 2 \\\\ 
        0 & \\text{{otherwise}} 
        \\end{{cases}}$$

        #### Finding the Expectation

        $$E[X] = \\int_{{-\\infty}}^{{\\infty}} x \\cdot f(x) \\, dx = \\int_0^2 x \\cdot \\frac{{3}}{{8}}(4x - 2x^2) \\, dx$$

        $$E[X] = \\frac{{3}}{{8}} \\int_0^2 (4x^2 - 2x^3) \\, dx = \\frac{{3}}{{8}} \\left[ \\frac{{4x^3}}{{3}} - \\frac{{2x^4}}{{4}} \\right]_0^2$$

        $$E[X] = \\frac{{3}}{{8}} \\left[ \\frac{{4 \\cdot 2^3}}{{3}} - \\frac{{2 \\cdot 2^4}}{{4}} - 0 \\right] = \\frac{{3}}{{8}} \\left[ \\frac{{32}}{{3}} - 4 \\right]$$

        $$E[X] = \\frac{{3}}{{8}} \\cdot \\frac{{32 - 12}}{{3}} = \\frac{{3}}{{8}} \\cdot \\frac{{20}}{{3}} = \\frac{{20}}{{8}} = {E_X}$$

        #### Computing the Variance

        We first need $E[X^2]$:

        $$E[X^2] = \\int_{{-\\infty}}^{{\\infty}} x^2 \\cdot f(x) \\, dx = \\int_0^2 x^2 \\cdot \\frac{{3}}{{8}}(4x - 2x^2) \\, dx$$

        $$E[X^2] = \\frac{{3}}{{8}} \\int_0^2 (4x^3 - 2x^4) \\, dx = \\frac{{3}}{{8}} \\left[ \\frac{{4x^4}}{{4}} - \\frac{{2x^5}}{{5}} \\right]_0^2$$

        $$E[X^2] = \\frac{{3}}{{8}} \\left[ 4 - \\frac{{2 \\cdot 32}}{{5}} - 0 \\right] = \\frac{{3}}{{8}} \\left[ 4 - \\frac{{64}}{{5}} \\right]$$

        $$E[X^2] = \\frac{{3}}{{8}} \\cdot \\frac{{20 - 64/5}}{{1}} = {E_X2}$$

        Now we calculate variance using the formula $Var(X) = E[X^2] - (E[X])^2$:

        $$\\text{{Var}}(X) = E[X^2] - (E[X])^2 = {E_X2} - ({E_X})^2 = {Var_X}$$

        This gives us a standard deviation of $\\sqrt{{\\text{{Var}}(X)}} = {Std_X}$.
        """
    )
    mo.vstack([_img, _calculations])
    return (
        E_X,
        E_X2,
        E_X_val,
        Std_X,
        Std_X_val,
        Var_X,
        Var_X_val,
        create_expectation_variance_vis,
        symbolic_stats_calc,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding

        Select which of these statements about continuous distributions you think are correct:

        /// details | The PDF of a continuous random variable can have values greater than 1
        ✅ Correct! Since the PDF represents density (not probability), it can exceed 1 as long as the total area under the curve equals 1.
        ///

        /// details | For a continuous distribution, $P(X = a) > 0$ for any value $a$ in the support
        ❌ Incorrect! For continuous random variables, the probability of the random variable taking any specific exact value is always 0. That is, $P(X = a) = 0$ for any value $a$.
        ///

        /// details | The area under a PDF curve between $a$ and $b$ equals the probability $P(a \leq X \leq b)$
        ✅ Correct! The area under the PDF curve over an interval gives the probability that the random variable falls within that interval.
        ///

        /// details | The CDF function $F(x)$ is always equal to $\int_{-\infty}^{x} f(t) \, dt$
        ✅ Correct! The CDF at point $x$ is the integral of the PDF from negative infinity to $x$.
        ///

        /// details | For a continuous random variable, $F(x)$ ranges from 0 to the maximum value in the support of the random variable
        ❌ Incorrect! The CDF $F(x)$ ranges from 0 to 1, representing probabilities. It approaches 1 (not the maximum value in the support) as $x$ approaches infinity.
        ///

        /// details | To calculate the variance of a continuous random variable, we use the formula $\text{Var}(X) = E[X^2] - (E[X])^2$
        ✅ Correct! This formula applies to both discrete and continuous random variables.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        Moving from discrete to continuous thinking is a big conceptual leap, but it opens up powerful ways to model real-world phenomena.

        In this notebook, we've seen how continuous random variables let us model quantities that can take any real value. Instead of dealing with probabilities at specific points (which are actually zero!), we work with probability density functions (PDFs) and find probabilities by calculating areas under curves.

        Some key points to remember:

        • PDFs give us relative likelihood, not actual probabilities - that's why they can exceed 1
        • The probability between two points is the area under the PDF curve
        • CDFs offer a convenient shortcut to find probabilities without integrating
        • Expectation and variance work similarly to discrete variables, just with integrals instead of sums
        • Constants in PDFs are determined by ensuring the total probability equals 1

        This foundation will serve you well as we explore specific continuous distributions like normal, exponential, and beta in future notebooks. These distributions are the workhorses of probability theory and statistics, appearing everywhere from quality control to financial modeling.

        One final thought: continuous distributions are beautiful mathematical objects, but remember they're just models. Real-world data is often discrete at some level, but continuous distributions provide elegant approximations that make calculations more tractable.
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
    import sympy
    from scipy import integrate as scipy
    import polars as pl
    import altair as alt
    from wigglystuff import TangleSlider
    return TangleSlider, alt, np, pl, plt, scipy, stats, sympy


@app.cell(hide_code=True)
def _():
    import io
    import base64
    from matplotlib.figure import Figure

    # Helper function to convert mpl figure to an image format mo.image can handle
    def fig_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        data = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        return data
    return Figure, base64, fig_to_image, io


@app.cell(hide_code=True)
def _(np, plt):
    def create_pdf_visualization(a, b, distribution='uniform'):
        fig, ax = plt.subplots(figsize=(10, 6))

        # x-values
        x = np.linspace(-0.5, 5.5, 1000)

        # Various PDFs to visualize
        if distribution == 'uniform':
            # Uniform distribution from 0 to 5
            y = np.where((x >= 0) & (x <= 5), 0.2, 0)
            title = f"Uniform PDF from 0 to 5"

        elif distribution == 'triangular':
            # Triangular distribution peaked at 2.5
            y = np.where(x <= 2.5, x/6.25, (5-x)/6.25)  # peak at 2.5
            y = np.where((x < 0) | (x > 5), 0, y)
            title = f"Triangular PDF from 0 to 5"

        elif distribution == 'exponential':
            lambda_param = 0.5
            y = np.where(x >= 0, lambda_param * np.exp(-lambda_param * x), 0)
            title = f"Exponential PDF with λ = {lambda_param}"

        # Plot PDF
        ax.plot(x, y, 'b-', linewidth=2, label='PDF $f(x)$')

        # Shade the area for the probability P(a ≤ X ≤ b)
        mask = (x >= a) & (x <= b)
        ax.fill_between(x[mask], y[mask], color='orange', alpha=0.5)

        # Calculate the probability
        dx = x[1] - x[0]
        probability = np.sum(y[mask]) * dx

        # vertical lines at a and b
        ax.axvline(x=a, color='r', linestyle='--', alpha=0.7, 
                   label=f'a = {a:.1f}')
        ax.axvline(x=b, color='g', linestyle='--', alpha=0.7, 
                   label=f'b = {b:.1f}')

        # horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density $f(x)$')
        ax.set_title(title)
        ax.legend(loc='upper right')

        # relevant annotations
        ax.annotate(f'$P({a:.1f} \leq X \leq {b:.1f}) = {probability:.4f}$', 
                    xy=(0.5, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                    horizontalalignment='center', fontsize=12)

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.gca()
        return fig, probability
    return (create_pdf_visualization,)


@app.cell(hide_code=True)
def _(np, plt, sympy):
    def create_example_pdf_visualization():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # x-values
        x = np.linspace(-0.5, 2.5, 1000)

        # PDF function
        C = 3/8
        pdf = np.where((x > 0) & (x < 2), C * (4*x - 2*x**2), 0)

        # CDF
        cdf = np.zeros_like(x)
        for i, val in enumerate(x):
            if val <= 0:
                cdf[i] = 0
            elif val >= 2:
                cdf[i] = 1
            else:
                # Analytical form: C*(2x^2 - 2x^3/3)
                cdf[i] = C * (2*val**2 - (2*val**3)/3)

        # PDF Plot
        ax1.plot(x, pdf, 'b-', linewidth=2)
        ax1.set_title('PDF: $f(x) = \\frac{3}{8}(4x - 2x^2)$ for $0 < x < 2$')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability Density')
        ax1.grid(alpha=0.3)

        # Highlight the area for P(X > 1)
        mask = (x > 1) & (x < 2)
        ax1.fill_between(x[mask], pdf[mask], color='orange', alpha=0.5, 
                        label='P(X > 1) = 0.5')

        # Add vertical line at x=1
        ax1.axvline(x=1, color='r', linestyle='--', alpha=0.7)
        ax1.legend()

        # CDF Plot
        ax2.plot(x, cdf, 'r-', linewidth=2)
        ax2.set_title('CDF: $F(x)$ for the Example Distribution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Cumulative Probability')
        ax2.grid(alpha=0.3)

        # Mark appropriate (F(1) & F(2)) points)
        ax2.plot(1, cdf[np.abs(x-1).argmin()], 'ro', markersize=8)
        ax2.plot(2, cdf[np.abs(x-2).argmin()], 'ro', markersize=8)

        # annotations
        F_1 = C * (2*1**2 - (2*1**3)/3)  # F(1)
        ax2.annotate(f'F(1) = {F_1:.3f}', xy=(1, F_1), xytext=(1.1, 0.4),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

        ax2.annotate(f'F(2) = 1', xy=(2, 1), xytext=(1.7, 0.8),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

        ax2.annotate(f'P(X > 1) = 1 - F(1) = {1-F_1:.3f}', xy=(1.5, 0.7), 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.2))

        # common x-limits
        for ax in [ax1, ax2]:
            ax.set_xlim(-0.25, 2.25)

        plt.tight_layout()
        plt.gca()
        return fig

    def symbolic_calculation():
        x = sympy.symbols('x')
        C = sympy.Rational(3, 8)

        # PDF defn
        pdf_expr = C * (4*x - 2*x**2)

        # Verify PDF integrates to 1
        total_prob = sympy.integrate(pdf_expr, (x, 0, 2))

        # Calculate P(X > 1)
        prob_gt_1 = sympy.integrate(pdf_expr, (x, 1, 2))

        return f"""Symbolic calculation verification:

        1. Total probability: ∫₀² {C}(4x - 2x²) dx = {total_prob}
        2. P(X > 1): ∫₁² {C}(4x - 2x²) dx = {prob_gt_1}
        """
    return create_example_pdf_visualization, symbolic_calculation


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "scipy==1.15.2",
#     "numpy==2.2.4",
#     "plotly==5.18.0",
#     "wigglystuff==0.1.13",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Central Limit Theorem")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Central Limit Theorem

    _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part4/clt/), by Stanford professor Chris Piech._

    The central limit theorem is honestly mind-blowing — it's like magic that no matter what distribution you start with, the sampling distribution of means approaches a normal distribution as sample size increases.

    Mathematically, if we have:

    $X_1, X_2, \ldots, X_n$ as independent, identically distributed random variables with:

    - Mean: $\mu$
    - Variance: $\sigma^2 < \infty$

    Then as $n \to \infty$:

    $$\sqrt{n}\left(\frac{1}{n}\sum_{i=1}^{n}X_i - \mu\right) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

    > _Note:_ The above LaTeX derivation is included as a reference. Credit for this formulation goes to the original source linked at the top of the notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Central Limit Theorem Statement

    There are two ways to state the central limit theorem:

    ### Sum Version

    Let $X_1, X_2, \dots, X_n$ be independent and identically distributed random variables. The sum of these random variables approaches a normal distribution as $n \rightarrow \infty$:

    $\sum_{i=1}^{n}X_i \sim \mathcal{N}(n \cdot \mu, n \cdot \sigma^2)$

    Where $\mu = E[X_i]$ and $\sigma^2 = \text{Var}(X_i)$. Since each $X_i$ is identically distributed, they share the same expectation and variance.

    ### Average Version

    Let $X_1, X_2, \dots, X_n$ be independent and identically distributed random variables. The average of these random variables approaches a normal distribution as $n \rightarrow \infty$:

    $\frac{1}{n}\sum_{i=1}^{n}X_i \sim \mathcal{N}(\mu, \frac{\sigma^2}{n})$

    Where $\mu = E[X_i]$ and $\sigma^2 = \text{Var}(X_i)$.

    The CLT is incredible because it applies to almost any distribution (as long as it has a finite mean and variance), regardless of its shape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Central Limit Theorem Intuition

    Let's explore what happens when you add random variables together. For example, what if we add 100 different uniform random variables?

    ```python
    from random import random

    def add_100_uniforms():
       total = 0
       for i in range(100):
           # returns a sample from uniform(0, 1)
           x_i = random()
           total += x_i
       return total
    ```

    The value returned by this function will be a random variable. Click the button below to run the function and observe the resulting value of total:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(label="Run add_100_uniforms()")

    run_button.center()
    return (run_button,)


@app.cell(hide_code=True)
def _(mo, random, run_button):
    def add_100_uniforms():
        total = 0
        for i in range(100):
            # returns a sample from uniform(0, 1)
            x_i = random.random()    
            total += x_i
        return total

    # Display the result when the button is clicked
    if run_button.value:
        uniform_result = add_100_uniforms()
        display = mo.md(f"**total**: {uniform_result:.5f}")
    else:
        display = mo.md("")

    display
    return (add_100_uniforms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What does total look like as a distribution? Let's calculate total many times and visualize the histogram of values it produces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Simulation control
    run_simulation_button = mo.ui.button(
        value=0, 
        on_click=lambda value: value + 1, 
        label="Run 10,000 more samples", 
        kind="warn"
    )

    run_simulation_button.center()
    return (run_simulation_button,)


@app.cell(hide_code=True)
def _(add_100_uniforms, go, mo, np, run_simulation_button, stats, time):
    # store the results
    def get_simulation_results():
        if not hasattr(get_simulation_results, "results"):
            get_simulation_results.results = []
            get_simulation_results.last_button_value = -1  # track button clicks
        return get_simulation_results

    # grab the results
    sim_storage = get_simulation_results()
    simulation_results = sim_storage.results

    # Check if button was clicked (value changed)
    if run_simulation_button.value != sim_storage.last_button_value:
        # Update the last seen button value
        sim_storage.last_button_value = run_simulation_button.value

        with mo.status.spinner(title="Running simulation...") as progress_status:
            sim_count = 10000
            new_results = []
            for _ in mo.status.progress_bar(range(sim_count)):
                sim_result = add_100_uniforms()
                new_results.append(sim_result)
                time.sleep(0.0001)  # tiny pause

            simulation_results.extend(new_results)

            progress_status.update(f"✅ Added {sim_count:,} samples (total: {len(simulation_results):,})")

    if simulation_results:
        # Numbers
        mean = np.mean(simulation_results)
        std_dev = np.std(simulation_results)

        theoretical_mean = 100 * 0.5  # = 50
        theoretical_variance = 100 * (1/12)  # = 8.33...
        theoretical_std = np.sqrt(theoretical_variance)  # ≈ 2.89

        # should be 10k times the click number (mainly for the y-axis label)
        total_samples = run_simulation_button.value * 10000

        fig = go.Figure()

        # histogram of samples
        fig.add_trace(go.Histogram(
            x=simulation_results,
            histnorm='probability density',
            name='Sum Distribution',
            marker_color='royalblue',
            opacity=0.7
        ))

        x_vals = np.linspace(min(simulation_results), max(simulation_results), 1000)
        y_vals = stats.norm.pdf(x_vals, theoretical_mean, theoretical_std)

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Normal approximation',
            line=dict(color='red', width=2)
        ))

        fig.add_vline(
            x=mean, 
            line_dash="dash", 
            line_width=1.5,
            line_color="green",
            annotation_text=f"Sample Mean: {mean:.2f}",
            annotation_position="top right"
        )

        # some notes
        fig.add_annotation(
            x=0.02, y=0.95,
            xref="paper", yref="paper",
            text=f"Sum of 100 Uniform(0,1) variables<br>" +
                 f"Sample size: {total_samples:,}<br>" +
                 f"Sample mean: {mean:.2f} (expected: {theoretical_mean})<br>" +
                 f"Sample std: {std_dev:.2f} (expected: {theoretical_std:.2f})<br>" +
                 f"According to CLT: Normal({theoretical_mean}, {theoretical_variance:.2f})",
            showarrow=False,
            align="left",
            bgcolor="white",
            opacity=0.8
        )

        fig.update_layout(
            title=f'Distribution of Sum of 100 Uniforms (Click #{run_simulation_button.value})',
            xaxis_title='Values',
            yaxis_title=f'Probability Density ({total_samples:,} runs)',
            template='plotly_white',
            height=500
        )

        # show
        histogram = mo.ui.plotly(fig)
    else:
        histogram = mo.md("Click the button to run the simulation!")

    # display
    histogram
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That is interesting! The sum of 100 independent uniforms looks normal. Is that a special property of uniforms? No! It turns out to work for almost any type of distribution (as long as the distribution has finite mean and variance).

    - Sum of 40 $X_i$ where $X_i \sim \text{Beta}(a = 5, b = 4)$? Normal.
    - Sum of 90 $X_i$ where $X_i \sim \text{Poisson}(\lambda = 4)$? Normal.
    - Sum of 50 dice-rolls? Normal.
    - Average of 10000 $X_i$ where $X_i \sim \text{Exp}(\lambda = 8)$? Normal.

    For any distribution, the sum or average of a sufficiently large number of independent, identically distributed random variables will be approximately normally distributed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continuity Correction

    When using the Central Limit Theorem with discrete random variables (like a Binomial or Poisson), we need to apply a continuity correction. This is because we're approximating a discrete distribution with a continuous one (normal).

    The continuity correction involves adjusting the boundaries in probability calculations by ±0.5 to account for the discrete nature of the original variable.

    You should use a continuity correction any time your normal is approximating a discrete random variable. The rules for a general continuity correction are the same as the rules for the [binomial-approximation continuity correction](http://marimo.app/https://github.com/marimo-team/learn/blob/main/probability/14_binomial_distribution.py).

    In our example above, where we added 100 uniforms, a continuity correction isn't needed because the sum of uniforms is continuous. However, in examples with dice or other discrete distributions, a continuity correction would be necessary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Examples

    Let's work through some practical examples to see how the Central Limit Theorem is applied.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 1: Dice Game

    > _Note:_ The following application demonstrates the practical use of the Central Limit Theorem. The mathematical derivation is based on concepts from ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/clt/) by Chris Piech.

    Let's solve a fun probability problem: You roll a 6-sided die 10 times and let $X$ represent the total value of all 10 dice: $X = X_1 + X_2 + \dots + X_{10}$. You win if $X \leq 25$ or $X \geq 45$. What's your probability of winning?

    For a single die roll $X_i$, we know:
    - $E[X_i] = 3.5$
    - $\text{Var}(X_i) = \frac{35}{12}$

    **Solution Approach:**

    This is where the Central Limit Theorem shines! Since we're summing 10 independent, identically distributed random variables, we can approximate this sum with a normal distribution $Y$:

    $Y \sim \mathcal{N}(10 \cdot E[X_i], 10 \cdot \text{Var}(X_i)) = \mathcal{N}(35, 29.2)$

    Now calculating our winning probability:

    $P(X \leq 25 \text{ or } X \geq 45) = P(X \leq 25) + P(X \geq 45)$

    Since we're approximating a discrete distribution with a continuous one, we apply a continuity correction:

    $\approx P(Y < 25.5) + P(Y > 44.5) = P(Y < 25.5) + [1 - P(Y < 44.5)]$

    Converting to standard normal form:

    $\approx \Phi\left(\frac{25.5 - 35}{\sqrt{29.2}}\right) + \left[1 - \Phi\left(\frac{44.5 - 35}{\sqrt{29.2}}\right)\right]$

    $\approx \Phi(-1.76) + [1 - \Phi(1.76)]$

    $\approx 0.039 + (1 - 0.961) \approx 0.078$

    So your chance of winning is about 7.8% — not great odds, but that's probability for you!
    """)
    return


@app.cell(hide_code=True)
def _(create_dice_game_visualization, fig_to_image, mo):
    # Display visualization
    dice_game_fig = create_dice_game_visualization()
    dice_game_image = mo.image(fig_to_image(dice_game_fig), width="100%")

    dice_explanation = mo.md(
        r"""
        **Understanding the Visualization:**

        This graph shows our dice game in action. The blue bars represent the exact probability distribution for summing 10 dice, while the red curve shows our normal approximation from the Central Limit Theorem.

        I've highlighted the winning regions in orange:
        - The left region where $X \leq 25$
        - The right region where $X \geq 45$

        Together these regions cover about 7.8% of the total probability.

        What's fascinating here is how closely the normal curve approximates the actual discrete distribution — this is the Central Limit Theorem working its magic, even with just 10 random variables.
        """
    )

    mo.vstack([dice_game_image, dice_explanation])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 2: Algorithm Runtime Estimation

    > _Note:_ The following derivation demonstrates the practical application of the Central Limit Theorem for experimental design. The mathematical approach is based on concepts from ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/clt/) by Chris Piech.

    Here's a practical problem I encounter in performance testing: You've developed a new algorithm and want to measure its average runtime. You know the variance is $\sigma^2 = 4 \text{ sec}^2$, but need to estimate the true mean runtime $t$.

    The question: How many test runs do you need to be 95% confident your estimated mean is within ±0.5 seconds of the true value?

    Let $X_i$ represent the runtime of the $i$-th test (for $1 \leq i \leq n$).

    **Solution:**

    We need to find $n$ such that:

    $0.95 = P\left(-0.5 \leq \frac{\sum_{i=1}^n X_i}{n} - t \leq 0.5\right)$

    The Central Limit Theorem tells us that as $n$ increases, the sample mean approaches a normal distribution. Let's standardize this to work with the standard normal distribution:

    $Z = \frac{\left(\sum_{i=1}^n X_i\right) - n\mu}{\sigma \sqrt{n}} = \frac{\left(\sum_{i=1}^n X_i\right) - nt}{2 \sqrt{n}}$

    Rewriting our probability constraint in terms of $Z$:

    $0.95 = P\left(-0.5 \leq \frac{\sum_{i=1}^n X_i}{n} - t \leq 0.5\right) = P\left(\frac{-0.5 \sqrt{n}}{2} \leq Z \leq \frac{0.5 \sqrt{n}}{2}\right)$

    Using the properties of the standard normal CDF:

    $0.95 = \Phi\left(\frac{\sqrt{n}}{4}\right) - \Phi\left(-\frac{\sqrt{n}}{4}\right) = 2\Phi\left(\frac{\sqrt{n}}{4}\right) - 1$

    Solving for $\Phi\left(\frac{\sqrt{n}}{4}\right)$:

    $0.975 = \Phi\left(\frac{\sqrt{n}}{4}\right)$

    Using the inverse CDF:

    $\Phi^{-1}(0.975) = \frac{\sqrt{n}}{4}$

    $1.96 = \frac{\sqrt{n}}{4}$

    $n = 61.4$

    Rounding up, we need 62 test runs to achieve our desired confidence interval — a practical result we can immediately apply to our testing protocol.
    """)
    return


@app.cell(hide_code=True)
def _(create_algorithm_runtime_visualization, fig_to_image, mo):
    # Display visualization
    runtime_fig = create_algorithm_runtime_visualization()
    runtime_image = mo.image(fig_to_image(runtime_fig), width="100%")

    runtime_explanation = mo.md(
        r"""
        **Visualization Explanation:**

        The graph illustrates how the standard error of the mean (SEM) decreases as the number of trials increases. The standard error is calculated as $\frac{\sigma}{\sqrt{n}}$.

        - When we conduct 62 trials, the standard error is approximately 0.254 seconds.
        - With a 95% confidence level, this gives us a margin of error of about ±0.5 seconds (1.96 × 0.254 ≈ 0.5).
        - The shaded region shows how the confidence interval narrows as the number of trials increases.

        This demonstrates why 62 trials are sufficient to meet our requirements of estimating the mean runtime within ±0.5 seconds with 95% confidence.
        """
    )

    mo.vstack([runtime_image, runtime_explanation])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive CLT Explorer

    Let's explore how the Central Limit Theorem works with different underlying distributions. You can select a distribution type and see how the distribution of the sample mean changes as the sample size increases.
    """)
    return


@app.cell(hide_code=True)
def _(controls):
    controls
    return


@app.cell(hide_code=True)
def _(
    distribution_type,
    fig_to_image,
    mo,
    np,
    plt,
    run_explorer_button,
    sample_size,
    sim_count_slider,
    stats,
):
    # Run simulation when button is clicked
    if run_explorer_button.value:
        # Set distribution parameters based on selection
        if distribution_type.value == "uniform":
            dist_name = "Uniform(0, 1)"
            # For uniform(0,1): mean = 0.5, variance = 1/12
            true_mean = 0.5
            true_var = 1/12

            # generate samples
            def generate_sample():
                return np.random.uniform(0, 1, sample_size.value)

        elif distribution_type.value == "exponential":
            rate = 1.0
            dist_name = f"Exponential(λ={rate})"
            # For exponential(λ): mean = 1/λ, variance = 1/λ²
            true_mean = 1/rate
            true_var = 1/(rate**2)

            def generate_sample():
                return np.random.exponential(1/rate, sample_size.value)

        elif distribution_type.value == "binomial":
            n_param, p = 10, 0.3
            dist_name = f"Binomial(n={n_param}, p={p})"
            # For binomial(n,p): mean = np, variance = np(1-p)
            true_mean = n_param * p
            true_var = n_param * p * (1-p)

            def generate_sample():
                return np.random.binomial(n_param, p, sample_size.value)

        elif distribution_type.value == "poisson":
            rate = 3.0
            dist_name = f"Poisson(λ={rate})"
            # For poisson(λ): mean = λ, variance = λ
            true_mean = rate
            true_var = rate

            def generate_sample():
                return np.random.poisson(rate, sample_size.value)

        # Generate the simulation data using a spinner for progress
        with mo.status.spinner(title="Running simulation...") as explorer_progress:
            sample_means = []
            original_samples = []

            # Run simulations
            for _ in mo.status.progress_bar(range(sim_count_slider.value)):
                sample = generate_sample()

                # Store the first simulation's individual values for visualizing original distribution
                if len(original_samples) < 1000:  # limit to prevent memory issues
                    original_samples.extend(sample)

                # sample mean
                sample_means.append(np.mean(sample))

            # progress
            explorer_progress.update(f"✅ Completed {sim_count_slider.value:,} simulations")

            # Create visualization
            explorer_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Original distribution histogram
            ax1.hist(original_samples, bins=30, density=True, alpha=0.7, color='royalblue')
            ax1.set_title(f"Original Distribution: {dist_name}")

            # Theoretical mean line
            ax1.axvline(x=true_mean, color='red', linestyle='--', 
                        label=f'True Mean = {true_mean:.3f}')

            ax1.set_xlabel("Value")
            ax1.set_ylabel("Density")
            ax1.legend()

            # Sample means histogram and normal approximation
            sample_mean_mean = np.mean(sample_means)
            sample_mean_std = np.std(sample_means)
            expected_std = np.sqrt(true_var / sample_size.value)  # CLT prediction

            ax2.hist(sample_means, bins=30, density=True, alpha=0.7, color='forestgreen',
                    label=f'Sample Size = {sample_size.value}')

            # Normal approximation from CLT
            explorer_x = np.linspace(min(sample_means), max(sample_means), 1000)
            explorer_y = stats.norm.pdf(explorer_x, true_mean, expected_std)
            ax2.plot(explorer_x, explorer_y, 'r-', linewidth=2, label='CLT Normal Approximation')

            # Add mean line
            ax2.axvline(x=true_mean, color='purple', linestyle='--',
                       label=f'True Mean = {true_mean:.3f}')

            ax2.set_title(f"Distribution of Sample Means\n(CLT Prediction: N({true_mean:.3f}, {true_var/sample_size.value:.5f}))")
            ax2.set_xlabel("Sample Mean")
            ax2.set_ylabel("Density")
            ax2.legend()

            # Add CLT description
            explorer_fig.text(0.5, 0.01, 
                    f"Central Limit Theorem: As sample size increases, the distribution of sample means approaches\n" +
                    f"a normal distribution with mean = {true_mean:.3f} and variance = {true_var:.3f}/{sample_size.value} = {true_var/sample_size.value:.5f}",
                    ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout(rect=[0, 0.05, 1, 1])

            # Display plot
            explorer_image = mo.image(fig_to_image(explorer_fig), width="100%")
    else:
        explorer_image = mo.md("Click the 'Run Simulation' button to see how the Central Limit Theorem works.")

    explorer_image
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤔 Test Your Understanding

    /// details | What is the shape of the distribution of the sum of many independent random variables?
    The sum of many independent random variables approaches a normal distribution, regardless of the shape of the original distributions (as long as they have finite mean and variance). This is the essence of the Central Limit Theorem.
    ///

    /// details | If $X_1, X_2, \dots, X_{100}$ are IID random variables with $E[X_i] = 5$ and $Var(X_i) = 9$, what is the distribution of their sum?
    By the Central Limit Theorem, the sum $S = X_1 + X_2 + \dots + X_{100}$ follows a normal distribution with:

    - Mean: $E[S] = 100 \cdot E[X_i] = 100 \cdot 5 = 500$
    - Variance: $Var(S) = 100 \cdot Var(X_i) = 100 \cdot 9 = 900$

    Therefore, $S \sim \mathcal{N}(500, 900)$, or equivalently $S \sim \mathcal{N}(500, 30^2)$.
    ///

    /// details | When do you need to apply a continuity correction when using the Central Limit Theorem?
    You need to apply a continuity correction when you're using the normal approximation (through CLT) for a discrete random variable.

    For example, when approximating a binomial or Poisson distribution with a normal distribution, you should adjust boundaries by ±0.5 to account for the discrete nature of the original variable. This makes the approximation more accurate.
    ///

    /// details | If $X_1, X_2, \dots, X_{n}$ are IID random variables, how does the variance of their sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^{n}X_i$ change as $n$ increases?
    The variance of the sample mean decreases as the sample size $n$ increases. Specifically:

    $Var(\bar{X}) = \frac{Var(X_i)}{n}$

    This means that as we take more samples, the sample mean becomes more concentrated around the true mean of the distribution. This is why larger samples give more precise estimates.
    ///

    /// details | Why is the Central Limit Theorem so important in statistics?
    The Central Limit Theorem is foundational in statistics because:

    1. It allows us to make inferences about population parameters using sample statistics, regardless of the population's distribution.
    2. It explains why the normal distribution appears so frequently in natural phenomena.
    3. It enables the construction of confidence intervals and hypothesis tests for means, even when the underlying population distribution is unknown.
    4. It justifies many statistical methods that assume normality, even when working with non-normal data, provided the sample size is large enough.

    In essence, the CLT provides the theoretical justification for much of statistical inference.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix (helper code and functions)
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    from wigglystuff import TangleSlider
    return


@app.cell(hide_code=True)
def _():
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import io
    import base64
    import random
    import time
    import plotly.graph_objects as go
    import plotly.io as pio
    return base64, go, io, np, plt, random, stats, time


@app.cell(hide_code=True)
def _(base64, io):
    from matplotlib.figure import Figure

    # Helper function to convert matplotlib figures to images
    def fig_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    return (fig_to_image,)


@app.cell(hide_code=True)
def _(np, plt, stats):
    def create_dice_game_visualization():
        """Create a visualization for the dice game example."""
        # Parameters
        n_dice = 10
        dice_values = np.arange(1, 7)  # 1 to 6

        # Theoretical values
        single_die_mean = np.mean(dice_values)  # 3.5
        single_die_var = np.var(dice_values)    # 35/12

        # Sum distribution parameters
        sum_mean = n_dice * single_die_mean
        sum_var = n_dice * single_die_var
        sum_std = np.sqrt(sum_var)

        # Possible outcomes for the sum of 10 dice
        min_sum = n_dice * min(dice_values)  # 10
        max_sum = n_dice * max(dice_values)  # 60
        sum_values = np.arange(min_sum, max_sum + 1)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate PMF through convolution
        # For one die
        single_pmf = np.ones(6) / 6

        sum_pmf = single_pmf.copy()
        for _ in range(n_dice - 1):
            sum_pmf = np.convolve(sum_pmf, single_pmf)

        # Plot the PMF
        ax.bar(sum_values, sum_pmf, alpha=0.7, color='royalblue', label='Exact PMF')

        # Normal approximation
        x = np.linspace(min_sum - 5, max_sum + 5, 1000)
        y = stats.norm.pdf(x, sum_mean, sum_std)
        ax.plot(x, y, 'r-', linewidth=2, label='Normal Approximation')

        # Win conditions (x ≤ 25 or x ≥ 45)
        win_region_left = sum_values <= 25
        win_region_right = sum_values >= 45

        # Shade win regions
        ax.bar(sum_values[win_region_left], sum_pmf[win_region_left], 
               color='darkorange', alpha=0.7, label='Win Region')
        ax.bar(sum_values[win_region_right], sum_pmf[win_region_right], 
               color='darkorange', alpha=0.7)

        # Calculate win probability
        win_prob = np.sum(sum_pmf[win_region_left]) + np.sum(sum_pmf[win_region_right])

        # Add vertical lines for critical values
        ax.axvline(x=25.5, color='red', linestyle='--', linewidth=1.5, label='Critical Points')
        ax.axvline(x=44.5, color='red', linestyle='--', linewidth=1.5)

        # Add mean line
        ax.axvline(x=sum_mean, color='green', linestyle='--', linewidth=1.5, 
                   label=f'Mean = {sum_mean}')

        # Text box with relevant information
        textstr = '\n'.join((
            f'Number of dice: {n_dice}',
            f'Sum Mean: {sum_mean}',
            f'Sum Std Dev: {sum_std:.2f}',
            f'Win Probability: {win_prob:.4f}',
            f'CLT Approximation: {0.078:.4f}'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Formatting
        ax.set_xlabel('Sum of 10 Dice')
        ax.set_ylabel('Probability')
        ax.set_title('Central Limit Theorem: Dice Game Example')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.gca()
        return fig
    return (create_dice_game_visualization,)


@app.cell(hide_code=True)
def _(np, plt):
    def create_algorithm_runtime_visualization():
        """Create a visualization for the algorithm runtime example."""
        # Parameters
        variance = 4  # σ² = 4 sec²
        std_dev = np.sqrt(variance)  # σ = 2 sec
        confidence_level = 0.95
        z_score = 1.96  # for 95% confidence
        target_error = 0.5  # ±0.5 seconds

        # Calculate n needed for desired precision
        n_required = int(np.ceil((z_score * std_dev / target_error) ** 2))  # ≈ 62

        n_values = np.arange(1, 100)

        # standard error
        standard_errors = std_dev / np.sqrt(n_values)

        # margin of error
        margins_of_error = z_score * standard_errors

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # standard error vs sample size plot
        ax.plot(n_values, standard_errors, 'b-', linewidth=2, label='Standard Error of Mean')

        # Plot margin of error vs sample size
        ax.plot(n_values, margins_of_error, 'r--', linewidth=2, 
                label=f'{confidence_level*100}% Margin of Error')

        ax.axvline(x=n_required, color='green', linestyle='-', linewidth=1.5,
                   label=f'Required n = {n_required}')

        ax.axhline(y=target_error, color='purple', linestyle='--', linewidth=1.5,
                   label=f'Target Error = ±{target_error} sec')

        # Shade the region below target error
        ax.fill_between(n_values, 0, target_error, alpha=0.2, color='green')

        # intersection point
        ax.plot(n_required, target_error, 'ro', markersize=8)
        ax.annotate(f'({n_required}, {target_error} sec)',
                    xy=(n_required, target_error),
                    xytext=(n_required + 5, target_error + 0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

        # Text box with appropriate information
        textstr = '\n'.join((
            f'Algorithm Variance: {variance} sec²',
            f'Standard Deviation: {std_dev} sec',
            f'Confidence Level: {confidence_level*100}%',
            f'Z-score: {z_score}',
            f'Target Error: ±{target_error} sec',
            f'Required Sample Size: {n_required}'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Formatting
        ax.set_xlabel('Sample Size (n)')
        ax.set_ylabel('Error (seconds)')
        ax.set_title('Sample Size Determination for Algorithm Runtime Estimation')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 2)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig
    return (create_algorithm_runtime_visualization,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    The Central Limit Theorem is truly one of the most remarkable ideas in all of statistics. It tells us that when we add up many independent random variables, their sum will follow a normal distribution, regardless of what the original distributions looked like. This is why we see normal distributions so often in real life – many natural phenomena are the result of numerous small, independent factors adding up.

    What makes the CLT so powerful is its universality. Whether we're working with dice rolls, measurement errors, or stock market returns, as long as we have enough independent samples, their average or sum will be approximately normal. For sums, the distribution will be $\mathcal{N}(n\mu, n\sigma^2)$, and for averages, it's $\mathcal{N}(\mu, \frac{\sigma^2}{n})$.

    The CLT gives us the foundation for confidence intervals, hypothesis testing, and many other statistical tools. Without it, we'd have a much harder time making sense of data when we don't know the underlying population distribution. Just remember that if you're working with discrete distributions, you'll need to apply a continuity correction to get more accurate results.

    Next time you see a normal distribution in data, think about the Central Limit Theorem – it might be the reason behind that familiar bell curve!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # controls for the interactive explorer
    distribution_type = mo.ui.dropdown(
        options=["uniform", "exponential", "binomial", "poisson"],
        value="uniform",
        label="Distribution Type"
    )

    sample_size = mo.ui.slider(
        start =1,
        stop =100,
        step=1,
        value=30,
        label="Sample Size (n)"
    )

    sim_count_slider = mo.ui.slider(
        start =100,
        stop =10000,
        step=100,
        value=1000,
        label="Number of Simulations"
    )

    run_explorer_button = mo.ui.run_button(label="Run Simulation", kind="warn")

    controls = mo.hstack([
        mo.vstack([distribution_type, sample_size, sim_count_slider]),
        run_explorer_button
    ], justify='space-around')
    return (
        controls,
        distribution_type,
        run_explorer_button,
        sample_size,
        sim_count_slider,
    )


if __name__ == "__main__":
    app.run()

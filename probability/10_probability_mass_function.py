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
app = marimo.App(width="medium", app_title="Probability Mass Functions")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Probability Mass Functions

    _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/pmf/), by Stanford professor Chris Piech._

    PMFs are really important in discrete probability. They tell us how likely each possible outcome is for a discrete random variable.

    What's interesting about PMFs is that they can be represented in multiple ways - equations, graphs, or even empirical data. The core idea is simple: they map each possible value to its probability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Properties of a PMF

    For a function $p_X(x)$ to be a valid PMF:

    1. **Non-negativity**: probability can't be negative, so $p_X(x) \geq 0$ for all $x$
    2. **Unit total probability**: all probabilities sum to 1, i.e., $\sum_x p_X(x) = 1$

    The second property makes intuitive sense - a random variable must take some value, and the sum of all possibilities should be 100%.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PMFs as Graphs

    Let's start by looking at PMFs as graphs where the $x$-axis is the values that the random variable could take on and the $y$-axis is the probability of the random variable taking on said value.

    In the following example, we show two PMFs:

    - On the left: PMF for the random variable $X$ = the value of a single six-sided die roll
    - On the right: PMF for the random variable $Y$ = value of the sum of two dice rolls
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    # Single die PMF
    single_die_values = np.arange(1, 7)
    single_die_probs = np.ones(6) / 6

    # Two dice sum PMF
    two_dice_values = np.arange(2, 13)
    two_dice_probs = []

    for dice_sum in two_dice_values:
        if dice_sum <= 7:
            dice_prob = (dice_sum-1) / 36
        else:
            dice_prob = (13-dice_sum) / 36
        two_dice_probs.append(dice_prob)

    # Create side-by-side plots
    dice_fig, (dice_ax1, dice_ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Single die plot
    dice_ax1.bar(single_die_values, single_die_probs, width=0.4)
    dice_ax1.set_xticks(single_die_values)
    dice_ax1.set_xlabel('Value of die roll (x)')
    dice_ax1.set_ylabel('Probability: P(X = x)')
    dice_ax1.set_title('PMF of a Single Die Roll')
    dice_ax1.grid(alpha=0.3)

    # Two dice sum plot
    dice_ax2.bar(two_dice_values, two_dice_probs, width=0.4)
    dice_ax2.set_xticks(two_dice_values)
    dice_ax2.set_xlabel('Sum of two dice (y)')
    dice_ax2.set_ylabel('Probability: P(Y = y)')
    dice_ax2.set_title('PMF of Sum of Two Dice')
    dice_ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These graphs really show us how likely each value is when we roll the dice.

    looking at the right graph, when we see "6" on the $x$-axis with probability $\frac{5}{36}$ on the $y$-axis, that's telling us there's a $\frac{5}{36}$ chance of rolling a sum of 6 with two dice. or more formally: $P(Y = 6) = \frac{5}{36}$.

    Similarly, the value "2" has probability "$\frac{1}{36}$" - that's because there's only one way to get a sum of 2 (rolling 1 on both dice). and you'll notice there's no value for "1" since you can't get a sum of 1 with two dice - the minimum possible is 2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PMFs as Equations

    Here is the exact same information in equation form:

    For a single die roll $X$:
    $$P(X=x) = \frac{1}{6} \quad \text{ if } 1 \leq x \leq 6$$

    For the sum of two dice $Y$:
    $$P(Y=y) = \begin{cases}
    \frac{(y-1)}{36} & \text{ if } 2 \leq y \leq 7\\
    \frac{(13-y)}{36} & \text{ if } 8 \leq y \leq 12
    \end{cases}$$

    Let's implement the PMF for $Y$, the sum of two dice, in Python code:
    """)
    return


@app.cell
def _():
    def pmf_sum_two_dice(y_val):
        """Returns the probability that the sum of two dice is y"""
        if y_val < 2 or y_val > 12:
            return 0
        if y_val <= 7:
            return (y_val-1) / 36
        else:
            return (13-y_val) / 36

    # Test the function for a few values
    test_values = [1, 2, 7, 12, 13]
    for test_y in test_values:
        print(f"P(Y = {test_y}) = {pmf_sum_two_dice(test_y)}")
    return (pmf_sum_two_dice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's verify that our PMF satisfies the property that the sum of all probabilities equals 1:
    """)
    return


@app.cell
def _(pmf_sum_two_dice):
    # Verify that probabilities sum to 1
    verify_total_prob = sum(pmf_sum_two_dice(y_val) for y_val in range(2, 13))
    # Round to 10 decimal places to handle floating-point precision
    verify_total_prob_rounded = round(verify_total_prob, 10)
    print(f"Sum of all probabilities: {verify_total_prob_rounded}")
    return


@app.cell(hide_code=True)
def _(plt, pmf_sum_two_dice):
    # Create a visual verification
    verify_y_values = list(range(2, 13))
    verify_probabilities = [pmf_sum_two_dice(y_val) for y_val in verify_y_values]

    plt.figure(figsize=(10, 4))
    plt.bar(verify_y_values, verify_probabilities, width=0.4)
    plt.xticks(verify_y_values)
    plt.xlabel('Sum of two dice (y)')
    plt.ylabel('Probability: P(Y = y)')
    plt.title('PMF of Sum of Two Dice (Total Probability = 1)')
    plt.grid(alpha=0.3)

    # Add probability values on top of bars
    for verify_i, verify_prob in enumerate(verify_probabilities):
        plt.text(verify_y_values[verify_i], verify_prob + 0.001, f'{verify_prob:.3f}', ha='center')

    plt.gca()  # Return the current axes to ensure proper display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data to Histograms to Probability Mass Functions

    Here's something I find interesting — one way to represent a likelihood function is just through raw data. instead of mathematical formulas, we can actually approximate a PMF by collecting data points. let's see this in action by simulating lots of dice rolls and building an empirical PMF:
    """)
    return


@app.cell
def _(np):
    # Simulate rolling two dice many times
    sim_num_trials = 10000
    np.random.seed(42)  # For reproducibility

    # Generate random dice rolls
    sim_die1 = np.random.randint(1, 7, size=sim_num_trials)
    sim_die2 = np.random.randint(1, 7, size=sim_num_trials)

    # Calculate the sum
    sim_dice_sums = sim_die1 + sim_die2

    # Display a small sample of the data
    print(f"First 20 dice sums: {sim_dice_sums[:20]}")
    print(f"Total number of trials: {sim_num_trials}")
    return (sim_dice_sums,)


@app.cell(hide_code=True)
def _(collections, np, plt, sim_dice_sums):
    # Count the frequency of each sum
    sim_counter = collections.Counter(sim_dice_sums)

    # Sort the values
    sim_sorted_values = sorted(sim_counter.keys())

    # Calculate the empirical PMF
    sim_empirical_pmf = [sim_counter[x] / len(sim_dice_sums) for x in sim_sorted_values]

    # Calculate the theoretical PMF
    sim_theoretical_values = np.arange(2, 13)
    sim_theoretical_pmf = []
    for sim_y in sim_theoretical_values:
        if sim_y <= 7:
            sim_prob = (sim_y-1) / 36
        else:
            sim_prob = (13-sim_y) / 36
        sim_theoretical_pmf.append(sim_prob)

    # Create a comparison plot
    sim_fig, (sim_ax1, sim_ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Empirical PMF (normalized histogram)
    sim_ax1.bar(sim_sorted_values, sim_empirical_pmf, width=0.4)
    sim_ax1.set_xticks(sim_sorted_values)
    sim_ax1.set_xlabel('Sum of two dice')
    sim_ax1.set_ylabel('Empirical Probability')
    sim_ax1.set_title(f'Empirical PMF from {len(sim_dice_sums)} Trials')
    sim_ax1.grid(alpha=0.3)

    # Theoretical PMF
    sim_ax2.bar(sim_theoretical_values, sim_theoretical_pmf, width=0.4)
    sim_ax2.set_xticks(sim_theoretical_values)
    sim_ax2.set_xlabel('Sum of two dice')
    sim_ax2.set_ylabel('Theoretical Probability')
    sim_ax2.set_title('Theoretical PMF')
    sim_ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Let's also look at the raw counts (histogram)
    plt.figure(figsize=(10, 4))
    sim_counts = [sim_counter[x] for x in sim_sorted_values]
    plt.bar(sim_sorted_values, sim_counts, width=0.4)
    plt.xticks(sim_sorted_values)
    plt.xlabel('Sum of two dice')
    plt.ylabel('Frequency')
    plt.title('Histogram of Dice Sum Frequencies')
    plt.grid(alpha=0.3)

    # Add count values on top of bars
    for sim_i, sim_count in enumerate(sim_counts):
        plt.text(sim_sorted_values[sim_i], sim_count + 19, str(sim_count), ha='center')

    plt.gca()  # Return the current axes to ensure proper display
    return (sim_counter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When we normalize a histogram (divide each count by total sample size), we get a pretty good approximation of the true PMF. it's a simple yet powerful idea - count how many times each value appears, then divide by the total number of trials.

    let's make this concrete. say we want to estimate $P(Y=3)$ - the probability of rolling a sum of 3 with two dice. we just count how many 3's show up in our simulated rolls and divide by the total number of rolls:
    """)
    return


@app.cell
def _(sim_counter, sim_dice_sums):
    # Calculate P(Y=3) empirically
    sim_count_of_3 = sim_counter[3]
    sim_empirical_prob = sim_count_of_3 / len(sim_dice_sums)

    # Calculate P(Y=3) theoretically
    sim_theoretical_prob = 2/36  # There are 2 ways to get a sum of 3 out of 36 possible outcomes

    print(f"Count of sum=3: {sim_count_of_3}")
    print(f"Empirical P(Y=3): {sim_count_of_3}/{len(sim_dice_sums)} = {sim_empirical_prob:.4f}")
    print(f"Theoretical P(Y=3): 2/36 = {sim_theoretical_prob:.4f}")
    print(f"Difference: {abs(sim_empirical_prob - sim_theoretical_prob):.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see, with a large number of trials, the empirical PMF becomes a very good approximation of the theoretical PMF. This is an example of the [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers) in action.

    ## Interactive Example: Exploring PMFs

    Let's create an interactive tool to explore different PMFs:
    """)
    return


@app.cell
def _(dist_param1, dist_param2, dist_selection, mo):
    mo.hstack([dist_selection, dist_param1, dist_param2], justify="space-around")
    return


@app.cell(hide_code=True)
def _(mo):
    dist_selection = mo.ui.dropdown(
        options=[
            "bernoulli",
            "binomial",
            "geometric",
            "poisson"
        ],
        value="bernoulli",
        label="Select a distribution"
    )

    # Parameters for different distributions
    dist_param1 = mo.ui.slider(
        start=0.05, 
        stop=0.95, 
        step=0.05, 
        value=0.5, 
        label="p (success probability)"
    )

    dist_param2 = mo.ui.slider(
        start=1, 
        stop=20, 
        step=1, 
        value=10, 
        label="n (trials) or λ (rate)"
    )
    return dist_param1, dist_param2, dist_selection


@app.cell(hide_code=True)
def _(dist_param1, dist_param2, dist_selection, np, plt, stats):
    # Set up the plot based on the selected distribution
    if dist_selection.value == "bernoulli":
        # Bernoulli distribution
        dist_p = dist_param1.value
        dist_x_values = np.array([0, 1])
        dist_pmf_values = [1-dist_p, dist_p]
        dist_title = f"Bernoulli PMF (p = {dist_p:.2f})"
        dist_x_label = "Outcome (0 = Failure, 1 = Success)"
        dist_max_x = 1

    elif dist_selection.value == "binomial":
        # Binomial distribution
        dist_n = int(dist_param2.value)
        dist_p = dist_param1.value
        dist_x_values = np.arange(0, dist_n+1)
        dist_pmf_values = stats.binom.pmf(dist_x_values, dist_n, dist_p)
        dist_title = f"Binomial PMF (n = {dist_n}, p = {dist_p:.2f})"
        dist_x_label = "Number of Successes"
        dist_max_x = dist_n

    elif dist_selection.value == "geometric":
        # Geometric distribution
        dist_p = dist_param1.value
        dist_max_x = min(int(5/dist_p), 50)  # Limit the range for visualization
        dist_x_values = np.arange(1, dist_max_x+1)
        dist_pmf_values = stats.geom.pmf(dist_x_values, dist_p)
        dist_title = f"Geometric PMF (p = {dist_p:.2f})"
        dist_x_label = "Number of Trials Until First Success"

    else:  # Poisson
        # Poisson distribution
        dist_lam = dist_param2.value
        dist_max_x = int(dist_lam*3) + 1  # Reasonable range for visualization
        dist_x_values = np.arange(0, dist_max_x)
        dist_pmf_values = stats.poisson.pmf(dist_x_values, dist_lam)
        dist_title = f"Poisson PMF (λ = {dist_lam})"
        dist_x_label = "Number of Events"

    # Create the plot
    plt.figure(figsize=(10, 5))

    # For discrete distributions, use stem plot for clarity
    dist_markerline, dist_stemlines, dist_baseline = plt.stem(
        dist_x_values, dist_pmf_values, markerfmt='o', basefmt=' '
    )
    plt.setp(dist_markerline, markersize=6)
    plt.setp(dist_stemlines, linewidth=1.5)

    # Add a bar plot for better visibility
    plt.bar(dist_x_values, dist_pmf_values, alpha=0.3, width=0.4)

    plt.xlabel(dist_x_label)
    plt.ylabel("Probability: P(X = x)")
    plt.title(dist_title)
    plt.grid(alpha=0.3)

    # Calculate and display expected value and variance
    if dist_selection.value == "bernoulli":
        dist_mean = dist_p
        dist_variance = dist_p * (1-dist_p)
    elif dist_selection.value == "binomial":
        dist_mean = dist_n * dist_p
        dist_variance = dist_n * dist_p * (1-dist_p)
    elif dist_selection.value == "geometric":
        dist_mean = 1/dist_p
        dist_variance = (1-dist_p)/(dist_p**2)
    else:  # Poisson
        dist_mean = dist_lam
        dist_variance = dist_lam

    dist_std_dev = np.sqrt(dist_variance)

    # Add text with distribution properties
    dist_props_text = (
        f"Mean: {dist_mean:.3f}\n"
        f"Variance: {dist_variance:.3f}\n"
        f"Std Dev: {dist_std_dev:.3f}\n"
        f"Sum of probabilities: {sum(dist_pmf_values):.6f}"
    )

    plt.text(0.95, 0.95, dist_props_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.gca()  # Return the current axes to ensure proper display
    return dist_pmf_values, dist_x_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expected Value from a PMF

    The expected value (or mean) of a discrete random variable is calculated using its PMF:

    $$E[X] = \sum_x x \cdot p_X(x)$$

    This represents the long-run average value of the random variable.
    """)
    return


@app.cell
def _(dist_pmf_values, dist_x_values):
    def calc_expected_value(x_values, pmf_values):
        """Calculate the expected value of a discrete random variable."""
        return sum(x * p for x, p in zip(x_values, pmf_values))

    # Calculate expected value for the current distribution
    ev_dist_mean = calc_expected_value(dist_x_values, dist_pmf_values)

    print(f"Expected value: {ev_dist_mean:.4f}")
    return (ev_dist_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variance from a PMF

    The variance measures the spread or dispersion of a random variable around its mean:

    $$\text{Var}(X) = E[(X - E[X])^2] = \sum_x (x - E[X])^2 \cdot p_X(x)$$

    An alternative formula is:

    $$\text{Var}(X) = E[X^2] - (E[X])^2 = \sum_x x^2 \cdot p_X(x) - \left(\sum_x x \cdot p_X(x)\right)^2$$
    """)
    return


@app.cell
def _(dist_pmf_values, dist_x_values, ev_dist_mean, np):
    def calc_variance(x_values, pmf_values, mean_value):
        """Calculate the variance of a discrete random variable."""
        return sum((x - mean_value)**2 * p for x, p in zip(x_values, pmf_values))

    # Calculate variance for the current distribution
    var_dist_var = calc_variance(dist_x_values, dist_pmf_values, ev_dist_mean)
    var_dist_std_dev = np.sqrt(var_dist_var)

    print(f"Variance: {var_dist_var:.4f}")
    print(f"Standard deviation: {var_dist_std_dev:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PMF vs. CDF

    The **Cumulative Distribution Function (CDF)** is related to the PMF but gives the probability that the random variable $X$ is less than or equal to a value $x$:

    $$F_X(x) = P(X \leq x) = \sum_{k \leq x} p_X(k)$$

    While the PMF gives the probability mass at each point, the CDF accumulates these probabilities.
    """)
    return


@app.cell(hide_code=True)
def _(dist_pmf_values, dist_x_values, np, plt):
    # Calculate the CDF from the PMF
    cdf_dist_values = np.cumsum(dist_pmf_values)

    # Create a plot comparing PMF and CDF
    cdf_fig, (cdf_ax1, cdf_ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # PMF plot
    cdf_ax1.bar(dist_x_values, dist_pmf_values, width=0.4, alpha=0.7)
    cdf_ax1.set_xlabel('x')
    cdf_ax1.set_ylabel('P(X = x)')
    cdf_ax1.set_title('Probability Mass Function (PMF)')
    cdf_ax1.grid(alpha=0.3)

    # CDF plot - using step function with 'post' style for proper discrete representation
    cdf_ax2.step(dist_x_values, cdf_dist_values, where='post', linewidth=2, color='blue')
    cdf_ax2.scatter(dist_x_values, cdf_dist_values, s=50, color='blue')

    # Set appropriate limits for better visualization
    if len(dist_x_values) > 0:
        x_min = min(dist_x_values) - 0.5
        x_max = max(dist_x_values) + 0.5
        cdf_ax2.set_xlim(x_min, x_max)
        cdf_ax2.set_ylim(0, 1.05)  # CDF goes from 0 to 1

    cdf_ax2.set_xlabel('x')
    cdf_ax2.set_ylabel('P(X ≤ x)')
    cdf_ax2.set_title('Cumulative Distribution Function (CDF)')
    cdf_ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.gca()  # Return the current axes to ensure proper display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The graphs above illustrate the key difference between PMF and CDF:

    - **PMF (left)**: Shows the probability of the random variable taking each specific value: P(X = x)
    - **CDF (right)**: Shows the probability of the random variable being less than or equal to each value: P(X ≤ x)

    The CDF at any point is the sum of all PMF values up to and including that point. This is why the CDF is always non-decreasing and eventually reaches 1. For discrete distributions like this one, the CDF forms a step function that jumps at each value in the support of the random variable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test Your Understanding

    Choose what you believe are the correct options in the questions below:

    <details>
    <summary>If X is a discrete random variable with PMF p(x), then p(x) must always be less than 1</summary>
    ❌ False! While most values in a PMF are typically less than 1, a PMF can have p(x) = 1 for a specific value if the random variable always takes that value (with 100% probability).
    </details>

    <details>
    <summary>The sum of all probabilities in a PMF must equal exactly 1</summary>
    ✅ True! This is a fundamental property of any valid PMF. The total probability across all possible values must be 1, as the random variable must take some value.
    </details>

    <details>
    <summary>A PMF can be estimated from data by creating a normalized histogram</summary>
    ✅ True! Counting the frequency of each value and dividing by the total number of observations gives an empirical PMF.
    </details>

    <details>
    <summary>The expected value of a discrete random variable is always one of the possible values of the variable</summary>
    ❌ False! The expected value is a weighted average and may not be a value the random variable can actually take. For example, the expected value of a fair die roll is 3.5, which is not a possible outcome.
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical Applications of PMFs

    PMFs pop up everywhere - network engineers use them to model traffic patterns, reliability teams predict equipment failures, and marketers analyze purchase behavior. In finance, they help price options; in gaming, they're behind every dice roll. Machine learning algorithms like Naive Bayes rely on them, and they're essential for modeling rare events like genetic mutations or system failures.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    PMFs give us the probability picture for discrete random variables - they tell us how likely each value is, must be non-negative, and always sum to 1. We can write them as equations, draw them as graphs, or estimate them from data. They're the foundation for calculating expected values and variances, which we'll explore in our next notebook on Expectation, where we'll learn how to summarize random variables with a single, most "expected" value.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import collections
    return collections, np, plt, stats


if __name__ == "__main__":
    app.run()

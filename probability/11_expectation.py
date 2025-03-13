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

__generated_with = "0.11.19"
app = marimo.App(width="medium", app_title="Expectation")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Expectation

        _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/expectation/), by Stanford professor Chris Piech._

        A random variable is fully represented by its Probability Mass Function (PMF), which describes each value the random variable can take on and the corresponding probabilities. However, a PMF can contain a lot of information. Sometimes it's useful to summarize a random variable with a single value!

        The most common, and arguably the most useful, summary of a random variable is its **Expectation** (also called the expected value or mean).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Definition of Expectation

        The expectation of a random variable $X$, written $E[X]$, is the average of all the values the random variable can take on, each weighted by the probability that the random variable will take on that value.

        $$E[X] = \sum_x x \cdot P(X=x)$$

        Expectation goes by many other names: Mean, Weighted Average, Center of Mass, 1st Moment. All of these are calculated using the same formula.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Intuition Behind Expectation

        The expected value represents the long-run average value of a random variable over many independent repetitions of an experiment.

        For example, if you roll a fair six-sided die many times and calculate the average of all rolls, that average will approach the expected value of 3.5 as the number of rolls increases.

        Let's visualize this concept:
        """
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Simulate rolling a die many times
    exp_num_rolls = 1000
    exp_die_rolls = np.random.randint(1, 7, size=exp_num_rolls)

    # Calculate the running average
    exp_running_avg = np.cumsum(exp_die_rolls) / np.arange(1, exp_num_rolls + 1)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, exp_num_rolls + 1), exp_running_avg, label='Running Average')
    plt.axhline(y=3.5, color='r', linestyle='--', label='Expected Value (3.5)')
    plt.xlabel('Number of Rolls')
    plt.ylabel('Average Value')
    plt.title('Running Average of Die Rolls Approaching Expected Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xscale('log')  # Log scale to better see convergence

    # Add annotations
    plt.annotate('As the number of rolls increases,\nthe average approaches the expected value',
                xy=(exp_num_rolls, exp_running_avg[-1]), xytext=(exp_num_rolls/3, 4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    plt.gca()  # Return the current axes to ensure proper display
    return exp_die_rolls, exp_num_rolls, exp_running_avg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Properties of Expectation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "1. Linearity of Expectation": mo.md(
                r"""
                $$E[aX + b] = a \cdot E[X] + b$$

                Where $a$ and $b$ are constants (not random variables).

                This means that if you multiply a random variable by a constant, the expectation is multiplied by that constant. And if you add a constant to a random variable, the expectation increases by that constant.
                """
            ),
            "2. Expectation of the Sum of Random Variables": mo.md(
                r"""
                $$E[X + Y] = E[X] + E[Y]$$

                This is true regardless of the relationship between $X$ and $Y$. They can be dependent, and they can have different distributions. This also applies with more than two random variables:

                $$E\left[\sum_{i=1}^n X_i\right] = \sum_{i=1}^n E[X_i]$$
                """
            ),
            "3. Law of the Unconscious Statistician (LOTUS)": mo.md(
                r"""
                $$E[g(X)] = \sum_x g(x) \cdot P(X=x)$$

                This allows us to calculate the expected value of a function $g(X)$ of a random variable $X$ when we know the probability distribution of $X$ but don't explicitly know the distribution of $g(X)$.

                This theorem has the humorous name "Law of the Unconscious Statistician" (LOTUS) because it's so useful that you should be able to employ it unconsciously.
                """
            ),
            "4. Expectation of a Constant": mo.md(
                r"""
                $$E[a] = a$$

                Sometimes in proofs, you'll end up with the expectation of a constant (rather than a random variable). Since a constant doesn't change, its expected value is just the constant itself.
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Calculating Expectation

        Let's calculate the expected value for some common examples:

        ### Example 1: Fair Die Roll

        For a fair six-sided die, the PMF is:

        $$P(X=x) = \frac{1}{6} \text{ for } x \in \{1, 2, 3, 4, 5, 6\}$$

        The expected value is:

        $$E[X] = 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + 4 \cdot \frac{1}{6} + 5 \cdot \frac{1}{6} + 6 \cdot \frac{1}{6} = \frac{21}{6} = 3.5$$

        Let's implement this calculation in Python:
        """
    )
    return


@app.cell
def _():
    def calc_expectation_die():
        """Calculate the expected value of a fair six-sided die roll."""
        exp_die_values = range(1, 7)
        exp_die_probs = [1/6] * 6

        exp_die_expected = sum(x * p for x, p in zip(exp_die_values, exp_die_probs))
        return exp_die_expected

    exp_die_result = calc_expectation_die()
    print(f"Expected value of a fair die roll: {exp_die_result}")
    return calc_expectation_die, exp_die_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2: Sum of Two Dice

        Now let's calculate the expected value for the sum of two fair dice. First, we need the PMF:
        """
    )
    return


@app.cell
def _():
    def pmf_sum_two_dice(y_val):
        """Returns the probability that the sum of two dice is y."""
        # Count the number of ways to get sum y
        exp_count = 0
        for dice1 in range(1, 7):
            for dice2 in range(1, 7):
                if dice1 + dice2 == y_val:
                    exp_count += 1
        return exp_count / 36  # There are 36 possible outcomes (6Ã—6)

    # Test the function for a few values
    exp_test_values = [2, 7, 12]
    for exp_test_y in exp_test_values:
        print(f"P(Y = {exp_test_y}) = {pmf_sum_two_dice(exp_test_y)}")
    return exp_test_values, exp_test_y, pmf_sum_two_dice


@app.cell
def _(pmf_sum_two_dice):
    def calc_expectation_sum_two_dice():
        """Calculate the expected value of the sum of two dice."""
        exp_sum_two_dice = 0
        # Sum of dice can take on the values 2 through 12
        for exp_x in range(2, 13):
            exp_pr_x = pmf_sum_two_dice(exp_x)  # PMF gives P(sum is x)
            exp_sum_two_dice += exp_x * exp_pr_x
        return exp_sum_two_dice

    exp_sum_result = calc_expectation_sum_two_dice()

    # Round to 2 decimal places for display
    exp_sum_result_rounded = round(exp_sum_result, 2)

    print(f"Expected value of the sum of two dice: {exp_sum_result_rounded}")

    # Let's also verify this with a direct calculation
    exp_direct_calc = sum(x * pmf_sum_two_dice(x) for x in range(2, 13))
    exp_direct_calc_rounded = round(exp_direct_calc, 2)

    print(f"Direct calculation: {exp_direct_calc_rounded}")

    # Verify that this equals 7
    print(f"Is the expected value exactly 7? {abs(exp_sum_result - 7) < 1e-10}")
    return (
        calc_expectation_sum_two_dice,
        exp_direct_calc,
        exp_direct_calc_rounded,
        exp_sum_result,
        exp_sum_result_rounded,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Visualizing Expectation

        Let's visualize the expectation for the sum of two dice. The expected value is the "center of mass" of the PMF:
        """
    )
    return


@app.cell(hide_code=True)
def _(plt, pmf_sum_two_dice):
    # Create the visualization
    exp_y_values = list(range(2, 13))
    exp_probabilities = [pmf_sum_two_dice(y) for y in exp_y_values]

    dice_fig, dice_ax = plt.subplots(figsize=(10, 5))
    dice_ax.bar(exp_y_values, exp_probabilities, width=0.4)
    dice_ax.axvline(x=7, color='r', linestyle='--', linewidth=2, label='Expected Value (7)')

    dice_ax.set_xticks(exp_y_values)
    dice_ax.set_xlabel('Sum of two dice (y)')
    dice_ax.set_ylabel('Probability: P(Y = y)')
    dice_ax.set_title('PMF of Sum of Two Dice with Expected Value')
    dice_ax.grid(alpha=0.3)
    dice_ax.legend()

    # Add probability values on top of bars
    for exp_i, exp_prob in enumerate(exp_probabilities):
        dice_ax.text(exp_y_values[exp_i], exp_prob + 0.001, f'{exp_prob:.3f}', ha='center')

    plt.tight_layout()
    plt.gca()
    return dice_ax, dice_fig, exp_i, exp_prob, exp_probabilities, exp_y_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Demonstrating the Properties of Expectation

        Let's demonstrate some of these properties with examples:
        """
    )
    return


@app.cell
def _(exp_die_result):
    # Demonstrate linearity of expectation (1)
    # E[aX + b] = a*E[X] + b

    # For a die roll X with E[X] = 3.5
    prop_a = 2
    prop_b = 10

    # Calculate E[2X + 10] using the property
    prop_expected_using_property = prop_a * exp_die_result + prop_b
    prop_expected_using_property_rounded = round(prop_expected_using_property, 2)

    print(f"Using linearity property: E[{prop_a}X + {prop_b}] = {prop_a} * E[X] + {prop_b} = {prop_expected_using_property_rounded}")

    # Calculate E[2X + 10] directly
    prop_expected_direct = sum((prop_a * x + prop_b) * (1/6) for x in range(1, 7))
    prop_expected_direct_rounded = round(prop_expected_direct, 2)

    print(f"Direct calculation: E[{prop_a}X + {prop_b}] = {prop_expected_direct_rounded}")

    # Verify they match
    print(f"Do they match? {abs(prop_expected_using_property - prop_expected_direct) < 1e-10}")
    return (
        prop_a,
        prop_b,
        prop_expected_direct,
        prop_expected_direct_rounded,
        prop_expected_using_property,
        prop_expected_using_property_rounded,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Law of the Unconscious Statistician (LOTUS)

        Let's use LOTUS to calculate $E[X^2]$ for a die roll, which will be useful when we study variance:
        """
    )
    return


@app.cell
def _():
    # Calculate E[X^2] for a die roll using LOTUS (3)
    lotus_die_values = range(1, 7)
    lotus_die_probs = [1/6] * 6

    # Using LOTUS: E[X^2] = sum(x^2 * P(X=x))
    lotus_expected_x_squared = sum(x**2 * p for x, p in zip(lotus_die_values, lotus_die_probs))
    lotus_expected_x_squared_rounded = round(lotus_expected_x_squared, 2)

    expected_x_squared = 3.5**2
    expected_x_squared_rounded = round(expected_x_squared, 2)

    print(f"E[X^2] for a die roll = {lotus_expected_x_squared_rounded}")
    print(f"(E[X])^2 for a die roll = {expected_x_squared_rounded}")
    return (
        expected_x_squared,
        expected_x_squared_rounded,
        lotus_die_probs,
        lotus_die_values,
        lotus_expected_x_squared,
        lotus_expected_x_squared_rounded,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// Note
        Note that E[X^2] != (E[X])^2
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interactive Example

        Let's explore how the expected value changes as we adjust the parameters of common probability distributions. This interactive visualization focuses specifically on the relationship between distribution parameters and expected values.

        Use the controls below to select a distribution and adjust its parameters. The graph will show how the expected value changes across a range of parameter values.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Create UI elements for distribution selection
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
    return (dist_selection,)


@app.cell(hide_code=True)
def _(dist_selection):
    dist_selection.center()
    return


@app.cell(hide_code=True)
def _(dist_description):
    dist_description
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Adjust Parameters""")
    return


@app.cell(hide_code=True)
def _(controls):
    controls
    return


@app.cell(hide_code=True)
def _(
    dist_selection,
    lambda_range,
    np,
    param_lambda,
    param_n,
    param_p,
    param_range,
    plt,
):
    # Calculate expected values based on the selected distribution
    if dist_selection.value == "bernoulli":
        # Get parameter range for visualization
        p_min, p_max = param_range.value
        param_values = np.linspace(p_min, p_max, 100)

        # E[X] = p for Bernoulli
        expected_values = param_values
        current_param = param_p.value
        current_expected = round(current_param, 2)
        x_label = "p (probability of success)"
        title = "Expected Value of Bernoulli Distribution"
        formula = "E[X] = p"

    elif dist_selection.value == "binomial":
        # Get parameter range for visualization
        p_min, p_max = param_range.value
        param_values = np.linspace(p_min, p_max, 100)

        # E[X] = np for Binomial
        n = int(param_n.value)
        expected_values = [n * p for p in param_values]
        current_param = param_p.value
        current_expected = round(n * current_param, 2)
        x_label = "p (probability of success)"
        title = f"Expected Value of Binomial Distribution (n={n})"
        formula = f"E[X] = n × p = {n} × p"

    elif dist_selection.value == "geometric":
        # Get parameter range for visualization
        p_min, p_max = param_range.value
        # Ensure p is not 0 for geometric distribution
        p_min = max(0.01, p_min)
        param_values = np.linspace(p_min, p_max, 100)

        # E[X] = 1/p for Geometric
        expected_values = [1/p for p in param_values]
        current_param = param_p.value
        current_expected = round(1 / current_param, 2)
        x_label = "p (probability of success)"
        title = "Expected Value of Geometric Distribution"
        formula = "E[X] = 1/p"

    else:  # Poisson
        # Get parameter range for visualization
        lambda_min, lambda_max = lambda_range.value
        param_values = np.linspace(lambda_min, lambda_max, 100)

        # E[X] = lambda for Poisson
        expected_values = param_values
        current_param = param_lambda.value
        current_expected = round(current_param, 2)
        x_label = "λ (rate parameter)"
        title = "Expected Value of Poisson Distribution"
        formula = "E[X] = λ"

    # Create the plot
    dist_fig, dist_ax = plt.subplots(figsize=(10, 6))

    # Plot the expected value function
    dist_ax.plot(param_values, expected_values, 'b-', linewidth=2, label="Expected Value Function")

    # Mark the current parameter value
    dist_ax.plot(current_param, current_expected, 'ro', markersize=10, label=f"Current Value: E[X] = {current_expected}")

    # Add a horizontal line from y-axis to the current point
    dist_ax.hlines(current_expected, param_values[0], current_param, colors='r', linestyles='dashed')

    # Add a vertical line from x-axis to the current point
    dist_ax.vlines(current_param, 0, current_expected, colors='r', linestyles='dashed')

    # Add shaded area under the curve
    dist_ax.fill_between(param_values, 0, expected_values, alpha=0.2, color='blue')

    dist_ax.set_xlabel(x_label, fontsize=12)
    dist_ax.set_ylabel("Expected Value: E[X]", fontsize=12)
    dist_ax.set_title(title, fontsize=14, fontweight='bold')
    dist_ax.grid(True, alpha=0.3)

    # Move legend to lower right to avoid overlap with formula
    dist_ax.legend(loc='lower right', fontsize=10)

    # Add formula text box in upper left
    dist_props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    dist_ax.text(0.02, 0.95, formula, transform=dist_ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dist_props)

    # Set reasonable y-axis limits based on the distribution
    if dist_selection.value == "geometric":
        max_y = min(50, 2/max(0.01, param_values[0]))
        dist_ax.set_ylim(0, max_y)
    elif dist_selection.value == "binomial":
        dist_ax.set_ylim(0, int(param_n.value) + 1)
    else:
        dist_ax.set_ylim(0, max(expected_values) * 1.1)

    # Add annotations for current value
    annotation_x = current_param + (param_values[-1] - param_values[0]) * 0.05
    annotation_y = current_expected

    # Adjust annotation position if it would go off the chart
    if annotation_x > param_values[-1] * 0.9:
        annotation_x = current_param - (param_values[-1] - param_values[0]) * 0.2

    dist_ax.annotate(
        f"Parameter: {current_param:.2f}\nE[X] = {current_expected}",
        xy=(current_param, current_expected),
        xytext=(annotation_x, annotation_y),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, alpha=0.7),
        bbox=dist_props
    )

    plt.tight_layout()
    plt.gca()
    return (
        annotation_x,
        annotation_y,
        current_expected,
        current_param,
        dist_ax,
        dist_fig,
        dist_props,
        expected_values,
        formula,
        lambda_max,
        lambda_min,
        max_y,
        n,
        p_max,
        p_min,
        param_values,
        title,
        x_label,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Expectation vs. Mode

        The expected value (mean) of a random variable is not always the same as its most likely value (mode). Let's explore this with an example:
        """
    )
    return


@app.cell(hide_code=True)
def _(np, plt, stats):
    # Create a skewed distribution
    skew_n = 10
    skew_p = 0.25

    # Binomial PMF
    skew_x_values = np.arange(0, skew_n+1)
    skew_pmf_values = stats.binom.pmf(skew_x_values, skew_n, skew_p)

    # Find the mode (most likely value)
    skew_mode = skew_x_values[np.argmax(skew_pmf_values)]

    # Calculate the expected value
    skew_expected = skew_n * skew_p
    skew_expected_rounded = round(skew_expected, 2)

    # Create the plot
    skew_fig, skew_ax = plt.subplots(figsize=(10, 5))
    skew_ax.bar(skew_x_values, skew_pmf_values, alpha=0.7, width=0.4)

    # Add vertical lines for mode and expected value
    skew_ax.axvline(x=skew_mode, color='g', linestyle='--', linewidth=2, 
                label=f'Mode = {skew_mode} (Most likely value)')
    skew_ax.axvline(x=skew_expected, color='r', linestyle='--', linewidth=2, 
                label=f'Expected Value = {skew_expected_rounded} (Mean)')

    # Add annotations to highlight the difference
    skew_ax.annotate('Mode', xy=(skew_mode, 0.05), xytext=(skew_mode-2.0, 0.1),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5), color='green')
    skew_ax.annotate('Expected Value', xy=(skew_expected, 0.05), xytext=(skew_expected+1, 0.15),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5), color='red')

    # Highlight the difference between mode and expected value
    if skew_mode != int(skew_expected):
        # Add a shaded region between mode and expected value
        min_x = min(skew_mode, skew_expected)
        max_x = max(skew_mode, skew_expected)
        skew_ax.axvspan(min_x, max_x, alpha=0.2, color='purple')

        # Add text explaining the difference
        mid_x = (skew_mode + skew_expected) / 2
        skew_ax.text(mid_x, max(skew_pmf_values) * 0.5, 
                 f"Difference: {abs(skew_mode - skew_expected_rounded):.2f}",
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

    skew_ax.set_xlabel('Number of Successes')
    skew_ax.set_ylabel('Probability')
    skew_ax.set_title(f'Binomial Distribution (n={skew_n}, p={skew_p})')
    skew_ax.grid(alpha=0.3)
    skew_ax.legend()

    plt.tight_layout()
    plt.gca()
    return (
        max_x,
        mid_x,
        min_x,
        skew_ax,
        skew_expected,
        skew_expected_rounded,
        skew_fig,
        skew_mode,
        skew_n,
        skew_p,
        skew_pmf_values,
        skew_x_values,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// NOTE
        For the sum of two dice we calculated earlier, we found the expected value to be exactly 7. In that case, 7 also happens to be the mode (most likely outcome) of the distribution. However, this is just a coincidence for this particular example!

        As we can see from the binomial distribution above, the expected value (2.50) and the mode (2) are often different values (this is common in skewed distributions). The expected value represents the "center of mass" of the distribution, while the mode represents the most likely single outcome.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Applications of Expectation

        Expected value has numerous applications across various fields:

        1. **Finance**: Expected return on investments, option pricing
        2. **Insurance**: Setting premiums based on expected claims
        3. **Gaming**: Calculating the expected winnings in games of chance
        4. **Machine Learning**: Loss function minimization
        5. **Operations Research**: Decision making under uncertainty
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding

        Choose what you believe are the correct options in the questions below:

        <details>
        <summary>The expected value of a random variable is always one of the possible values the random variable can take.</summary>
        ❌ False! The expected value is a weighted average and may not be a value the random variable can actually take. For example, the expected value of a fair die roll is 3.5, which is not a possible outcome.
        </details>

        <details>
        <summary>If X and Y are independent random variables, then E[XÂ·Y] = E[X]Â·E[Y].</summary>
        ✅ True! For independent random variables, the expectation of their product equals the product of their expectations.
        </details>

        <details>
        <summary>The expected value of a constant random variable (one that always takes the same value) is that constant.</summary>
        ✅ True! If X = c with probability 1, then E[X] = c.
        </details>

        <details>
        <summary>The expected value of the sum of two random variables is always the sum of their expected values, regardless of whether they are independent.</summary>
        ✅ True! This is the linearity of expectation property: E[X + Y] = E[X] + E[Y], which holds regardless of dependence.
        </details>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Practical Applications of Expectation

        Expected values show up everywhere - from investment decisions and insurance pricing to machine learning algorithms and game design. Engineers use them to predict system reliability, data scientists to understand customer behavior, and economists to model market outcomes. They're essential for risk assessment in project management and for optimizing resource allocation in operations research.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Key Takeaways

        Expectation gives us a single value that summarizes a random variable's central tendency - it's the weighted average of all possible outcomes, where the weights are probabilities. The linearity property makes expectations easy to work with, even for complex combinations of random variables. While a PMF gives the complete probability picture, expectation provides an essential summary that helps us make decisions under uncertainty. In our next notebook, we'll explore variance, which measures how spread out a random variable's values are around its expectation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Appendix (containing helper code)""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import collections
    return collections, np, plt, stats


@app.cell(hide_code=True)
def _(dist_selection, mo):
    # Parameter controls for probability-based distributions
    param_p = mo.ui.slider(
        start=0.01, 
        stop=0.99, 
        step=0.01, 
        value=0.5, 
        label="p (probability of success)",
        full_width=True
    )

    # Parameter control for binomial distribution
    param_n = mo.ui.slider(
        start=1, 
        stop=50, 
        step=1, 
        value=10, 
        label="n (number of trials)",
        full_width=True
    )

    # Parameter control for Poisson distribution
    param_lambda = mo.ui.slider(
        start=0.1, 
        stop=20, 
        step=0.1, 
        value=5, 
        label="λ (rate parameter)",
        full_width=True
    )

    # Parameter range sliders for visualization
    param_range = mo.ui.range_slider(
        start=0, 
        stop=1, 
        step=0.01, 
        value=[0, 1], 
        label="Parameter range to visualize",
        full_width=True
    )

    lambda_range = mo.ui.range_slider(
        start=0, 
        stop=20, 
        step=0.1, 
        value=[0, 20], 
        label="λ range to visualize",
        full_width=True
    )

    # Display appropriate controls based on the selected distribution
    if dist_selection.value == "bernoulli":
        controls = mo.hstack([param_p, param_range], justify="space-around")
    elif dist_selection.value == "binomial":
        controls = mo.hstack([param_p, param_n, param_range], justify="space-around")
    elif dist_selection.value == "geometric":
        controls = mo.hstack([param_p, param_range], justify="space-around")
    else:  # poisson
        controls = mo.hstack([param_lambda, lambda_range], justify="space-around")
    return controls, lambda_range, param_lambda, param_n, param_p, param_range


@app.cell(hide_code=True)
def _(dist_selection, mo):
    # Create distribution descriptions based on selection
    if dist_selection.value == "bernoulli":
        dist_description = mo.md(
            r"""
            **Bernoulli Distribution**

            A Bernoulli distribution models a single trial with two possible outcomes: success (1) or failure (0).

            - Parameter: $p$ = probability of success
            - Expected Value: $E[X] = p$
            - Example: Flipping a coin once (p = 0.5 for a fair coin)
            """
        )
    elif dist_selection.value == "binomial":
        dist_description = mo.md(
            r"""
            **Binomial Distribution**

            A Binomial distribution models the number of successes in $n$ independent trials.

            - Parameters: $n$ = number of trials, $p$ = probability of success
            - Expected Value: $E[X] = np$
            - Example: Number of heads in 10 coin flips
            """
        )
    elif dist_selection.value == "geometric":
        dist_description = mo.md(
            r"""
            **Geometric Distribution**

            A Geometric distribution models the number of trials until the first success.

            - Parameter: $p$ = probability of success
            - Expected Value: $E[X] = \frac{1}{p}$
            - Example: Number of coin flips until first heads
            """
        )
    else:  # poisson
        dist_description = mo.md(
            r"""
            **Poisson Distribution**

            A Poisson distribution models the number of events occurring in a fixed interval.

            - Parameter: $\lambda$ = average rate of events
            - Expected Value: $E[X] = \lambda$
            - Example: Number of emails received per hour
            """
        )
    return (dist_description,)


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # 🎲 Mathematical Probability: First Steps

In this notebook, we'll explore the
        fundamental approaches to understanding probability.

        ## What is Probability?

        Probability is a measure of the likelihood of an event occurring. It's a number between 
        0 (impossible) and 1 (certain).

        We'll explore two main approaches:

        - 🎯 Classical (Theoretical) Approach

        - 📊 Empirical (Experimental) Approach
        """
    )
    return


@app.cell
def _(approach):
    approach
    return


@app.cell
def _(mo):
    approach = mo.ui.tabs({
        "Classical": """
            ### Classical Probability
            
            The classical approach assumes:
            
            - All outcomes are equally likely
            
            - We can count all possible outcomes
            
            $P(Event) = \\frac{\\text{Favorable Outcomes}}{\\text{Total Outcomes}}$
            """,
        "Empirical": """
            ### Empirical Probability
            
            Based on actual experiments/observations:
            
            $P(Event) = \\frac{\\text{Number of times event occurs}}{\\text{Total number of trials}}$
            """
    })
    return (approach,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Interactive Example: Coin Flips 🪙

        Let's understand both approaches using a simple coin flip experiment.
        Try adjusting the number of flips and see how the empirical probability 
        compares to the theoretical probability of 0.5!
        """
    )
    return


@app.cell
def _(num_flips):
    num_flips
    return


@app.cell
def _(mo):
    num_flips = mo.ui.slider(
        value=100,
        start=10,
        stop=1000,
        step=10,
        label="Number of Coin Flips"
    )
    return (num_flips,)


@app.cell
def _(mo, num_flips, random):
    # Simulate coin flips
    flips = [random.choice(['Heads', 'Tails']) for _ in range(num_flips.value)]
    heads_count = flips.count('Heads')
    empirical_prob = heads_count / num_flips.value

    result = f"""
    ### Results after {num_flips.value} flips:

    - Heads: {heads_count}
    - Tails: {num_flips.value - heads_count}
    - Empirical Probability of Heads: {empirical_prob:.3f}
    - Theoretical Probability: 0.500
    - Difference: {abs(empirical_prob - 0.5):.3f}
    """

    mo.hstack([
        mo.md(result),
        mo.md(f"{'🎯 Close match!' if abs(empirical_prob - 0.5) < 0.1 else '🔄 Keep flipping!'}")
    ])
    return empirical_prob, flips, heads_count, result


@app.cell
def _(mo):
    mo.md(
        """
        ## 🎮 Interactive Challenge: The Marble Game

        You have a bag of marbles. Let's calculate probabilities using both approaches!
        """
    )
    return


@app.cell
def _(mo, red_marbles, total_marbles):
    mo.hstack([total_marbles, red_marbles])
    return


@app.cell
def _(mo):
    total_marbles = mo.ui.number(value=10, start=5, stop=20, label="Total Marbles")
    red_marbles = mo.ui.number(value=4, start=0, stop=20, label="Red Marbles")
    return red_marbles, total_marbles


@app.cell
def _(mo, red_marbles, total_marbles):
    if red_marbles.value > total_marbles.value:
        mo.callout(
            "⚠️ Red marbles cannot exceed total marbles!", 
            kind="error"
        )

    theoretical_prob = red_marbles.value / total_marbles.value

    _result = f"""
    ### Classical Probability Analysis

    Given:

    - Total marbles: {total_marbles.value}

    - Red marbles: {red_marbles.value}

    $P(\\text{{Red}}) = \\frac{{{red_marbles.value}}}{{{total_marbles.value}}} = {theoretical_prob:.3f}$
    """

    mo.md(_result)
    return (theoretical_prob,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 🤔 Key Differences

        Let's compare the Classical and Empirical approaches:
        """
    )
    return


@app.cell
def _(mo):
    comparison1 = mo.md(r"""
            ## Classical Approach:
                
                - Based on theoretical calculations
                
                - Assumes equally likely outcomes
                
                - Requires counting possibilities
                
                - Gives exact probabilities
                """
    )
    comparison2 = mo.md(r""" 
            ## Empirical Approach:
                - Based on actual experiments
                
                - No assumptions needed
                
                - Requires many trials
                
                - Gives approximate probabilities
                """
    )
    mo.hstack([comparison1, comparison2], justify="space-around")
    return comparison1, comparison2


@app.cell
def _(mo):
    quiz = mo.md("""
    ## ✍️ Quick Check

    Try this:

    1. A fair die is rolled. What's the classical probability of getting a 6?

    2. If you roll a die 100 times and get 20 sixes, what's the empirical probability?

    3. Which would you trust more in this case, and why?

    <details>
    <summary>Click for answers!</summary>

    1. Classical: P(6) = 1/6 ≈ 0.167 <br>
    2. Empirical: P(6) = 20/100 = 0.200 <br>
    3. Classical would be more reliable here because we know the die is fair!

    </details>
    """)

    mo.callout(quiz, kind="info")
    return (quiz,)


@app.cell
def _(mo):
    callout_text = mo.md("""
    ## 🎯 Your Probability Journey Begins!

    Next Steps:

    - Practice calculating probabilities both ways

    - Try more experiments with coins and dice

    - Think about real-world applications

    Coming up next: Set Theory Fundamentals! 📚✨
    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import random
    return (random,)


if __name__ == "__main__":
    app.run()

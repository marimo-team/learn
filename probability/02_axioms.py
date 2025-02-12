# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.2",
# ]
# ///

import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Axioms of Probability

        Probability theory is built on three fundamental axioms, known as the [Kolmogorov axioms](https://en.wikipedia.org/wiki/Probability_axioms). These axioms form 
        the mathematical foundation for all of probability theory[<sup>1</sup>](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/probability).

        Let's explore each axiom and understand why they make intuitive sense:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Three Axioms

        | Axiom | Mathematical Form | Meaning |
        |-------|------------------|----------|
        | **Axiom 1** | $0 \leq P(E) \leq 1$ | All probabilities are between 0 and 1 |
        | **Axiom 2** | $P(S) = 1$ | The probability of the sample space is 1 |
        | **Axiom 3** | $P(E \cup F) = P(E) + P(F)$ | For mutually exclusive events, probabilities add |

        where $S$ is the sample space (all possible outcomes), and $E$ and $F$ are events.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding Through Examples

        Let's explore these axioms using a simple experiment: rolling a fair six-sided die.
        We'll use this to demonstrate why each axiom makes intuitive sense.
        """
    )
    return


@app.cell
def _(event):
    event
    return


@app.cell(hide_code=True)
def _(mo):
    # Create an interactive widget to explore different events

    event = mo.ui.dropdown(

        options=[

            "Rolling an even number (2,4,6)",

            "Rolling an odd number (1,3,5)",

            "Rolling a prime number (2,3,5)",

            "Rolling less than 4 (1,2,3)",

            "Any possible roll (1,2,3,4,5,6)",

        ],

        value="Rolling an even number (2,4,6)",

        label="Select an event"

    )
    return (event,)


@app.cell(hide_code=True)
def _(event, mo, np, plt):
    # Define the probabilities for each event
    event_map = {
        "Rolling an even number (2,4,6)": [2, 4, 6],
        "Rolling an odd number (1,3,5)": [1, 3, 5],
        "Rolling a prime number (2,3,5)": [2, 3, 5],
        "Rolling less than 4 (1,2,3)": [1, 2, 3],
        "Any possible roll (1,2,3,4,5,6)": [1, 2, 3, 4, 5, 6],
    }

    # Get outcomes directly from the event value
    outcomes = event_map[event.value]
    prob = len(outcomes) / 6

    # Visualize the probability
    dice = np.arange(1, 7)
    colors = ['#1f77b4' if d in outcomes else '#d3d3d3' for d in dice]

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.bar(dice, np.ones_like(dice), color=colors)
    ax.set_xticks(dice)
    ax.set_ylim(0, 1.2)
    ax.set_title(f"P(Event) = {prob:.2f}")

    # Add explanation
    explanation = mo.md(f"""
    **Event**: {event.value}

    **Probability**: {prob:.2f}

    **Favorable outcomes**: {outcomes}

    This example demonstrates:

    - Axiom 1: The probability is between 0 and 1

    - Axiom 2: For the sample space, P(S) = 1

    - Axiom 3: The probability is the sum of individual outcome probabilities
    """)

    mo.hstack([plt.gcf(), explanation])
    return ax, colors, dice, event_map, explanation, fig, outcomes, prob


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Why These Axioms Matter

        These axioms are more than just rules - they provide the foundation for all of probability theory:

        1. **Non-negativity** (Axiom 1) makes intuitive sense: you can't have a negative number of occurrences 
           in any experiment.

        2. **Normalization** (Axiom 2) ensures that something must happen - the total probability must be 1.

        3. **Additivity** (Axiom 3) lets us build complex probabilities from simple ones, but only for events 
           that can't happen together (mutually exclusive events).

        From these simple rules, we can derive all the powerful tools of probability theory that are used in 
        statistics, machine learning, and other fields.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding

        Consider rolling two dice. Which of these statements follow from the axioms?

        <details>
        <summary>1. P(sum is 13) = 0</summary>

        ✅ Correct! This follows from Axiom 1. Since no combination of dice can sum to 13, 
        the probability must be non-negative but can be 0.
        </details>

        <details>
        <summary>2. P(sum is 7) + P(sum is not 7) = 1</summary>

        ✅ Correct! This follows from Axioms 2 and 3. These events are mutually exclusive and cover 
        the entire sample space.
        </details>

        <details>
        <summary>3. P(first die is 6 or second die is 6) = P(first die is 6) + P(second die is 6)</summary>

        ❌ Incorrect! This doesn't follow from Axiom 3 because the events are not mutually exclusive - 
        you could roll (6,6).
        </details>
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


if __name__ == "__main__":
    app.run()

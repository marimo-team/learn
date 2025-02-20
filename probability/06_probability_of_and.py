# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "matplotlib-venn"
# ]
# ///

import marimo

__generated_with = "0.11.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2
    return plt, venn2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Probability of And
        _This notebook is a computational companion to the book ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/prob_and/), by Stanford professor Chris Piech._

        When calculating the probability of both events occurring together, we need to consider whether the events are independent or dependent.
        Let's explore how to calculate $P(E \cap F)$, i.e. $P(E \text{ and } F)$, in different scenarios.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## And with Independent Events

        Two events $E$ and $F$ are **independent** if knowing one event occurred doesn't affect the probability of the other. 
        For independent events:

        $P(E \text{ and } F) = P(E) \cdot P(F)$

        For example:

        - Rolling a 6 on one die and getting heads on a coin flip
        - Drawing a heart from a deck, replacing it, and drawing another heart
        - Getting a computer error on Monday vs. Tuesday

        Here's a Python function to calculate probability for independent events:
        """
    )
    return


@app.cell
def _():
    def calc_independent_prob(p_e, p_f):
        return p_e * p_f

    # Example 1: Rolling a die and flipping a coin
    p_six = 1/6           # P(rolling a 6)
    p_heads = 1/2         # P(getting heads)
    p_both = calc_independent_prob(p_six, p_heads)
    print(f"Example 1: P(rolling 6 AND getting heads) = {p_six:.3f} × {p_heads:.3f} = {p_both:.3f}")
    return calc_independent_prob, p_both, p_heads, p_six


@app.cell
def _(calc_independent_prob):
    # Example 2: Two independent system components failing
    p_cpu_fail = 0.05     # P(CPU failure)
    p_disk_fail = 0.03    # P(disk failure)
    p_both_fail = calc_independent_prob(p_cpu_fail, p_disk_fail)
    print(f"Example 2: P(both CPU and disk failing) = {p_cpu_fail:.3f} × {p_disk_fail:.3f} = {p_both_fail:.3f}")
    return p_both_fail, p_cpu_fail, p_disk_fail


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## And with Dependent Events

        For dependent events, we use the **chain rule**:

        $P(E \text{ and } F) = P(E) \cdot P(F|E)$

        where $P(F|E)$ is the probability of $F$ occurring given that $E$ has occurred.

        For example:

        - Drawing two hearts without replacement
        - Getting two consecutive heads in poker
        - System failures in connected components

        Let's implement this calculation:
        """
    )
    return


@app.cell
def _():
    def calc_dependent_prob(p_e, p_f_given_e):
        return p_e * p_f_given_e

    # Example 1: Drawing two hearts without replacement
    p_first_heart = 13/52        # P(first heart)
    p_second_heart = 12/51       # P(second heart | first heart)
    p_both_hearts = calc_dependent_prob(p_first_heart, p_second_heart)
    print(f"Example 1: P(two hearts) = {p_first_heart:.3f} × {p_second_heart:.3f} = {p_both_hearts:.3f}")
    return calc_dependent_prob, p_both_hearts, p_first_heart, p_second_heart


@app.cell
def _(calc_dependent_prob):
    # Example 2: Drawing two aces without replacement
    p_first_ace = 4/52          # P(first ace)
    p_second_ace = 3/51         # P(second ace | first ace)
    p_both_aces = calc_dependent_prob(p_first_ace, p_second_ace)
    print(f"Example 2: P(two aces) = {p_first_ace:.3f} × {p_second_ace:.3f} = {p_both_aces:.3f}")
    return p_both_aces, p_first_ace, p_second_ace


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Multiple Events

        For multiple independent events:

        $P(E_1 \text{ and } E_2 \text{ and } \cdots \text{ and } E_n) = \prod_{i=1}^n P(E_i)$

        For dependent events:

        $P(E_1 \text{ and } E_2 \text{ and } \cdots \text{ and } E_n) = P(E_1) \cdot P(E_2|E_1) \cdot P(E_3|E_1,E_2) \cdots P(E_n|E_1,\ldots,E_{n-1})$

        Let's visualize these probabilities:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Interactive example""")
    return


@app.cell
def _(event_type):
    event_type
    return


@app.cell(hide_code=True)
def _(mo):
    event_type = mo.ui.dropdown(
        options=[
            "Independent AND (Die and Coin)",
            "Dependent AND (Sequential Cards)",
            "Multiple AND (System Components)"
        ],
        value="Independent AND (Die and Coin)",
        label="Select AND Probability Scenario"
    )
    return (event_type,)


@app.cell(hide_code=True)
def _(event_type, mo, plt, venn2):
    # Define the events and their probabilities
    events_data = {
        "Independent AND (Die and Coin)": {
            "sets": (0.33, 0.17, 0.08),  # (die, coin, intersection)
            "labels": ("Die\nP(6)=1/6", "Coin\nP(H)=1/2"),
            "title": "Independent Events: Rolling a 6 AND Getting Heads",
            "explanation": r"""
            ### Independent Events: Die Roll and Coin Flip

            $P(\text{Rolling 6}) = \frac{1}{6} \approx 0.17$

            $P(\text{Getting Heads}) = \frac{1}{2} = 0.5$

            $P(\text{6 and Heads}) = \frac{1}{6} \times \frac{1}{2} = \frac{1}{12} \approx 0.08$

            These events are independent because the outcome of the die roll 
            doesn't affect the coin flip, and vice versa.
            """,
        },
        "Dependent AND (Sequential Cards)": {
            "sets": (
                0.25,
                0.24,
                0.06,
            ),  # (first heart, second heart, intersection)
            "labels": ("First\nP(H₁)=13/52", "Second\nP(H₂|H₁)=12/51"),
            "title": "Dependent Events: Drawing Two Hearts",
            "explanation": r"""
            ### Dependent Events: Drawing Hearts

            $P(\text{First Heart}) = \frac{13}{52} = 0.25$

            $P(\text{Second Heart}|\text{First Heart}) = \frac{12}{51} \approx 0.24$

            $P(\text{Both Hearts}) = \frac{13}{52} \times \frac{12}{51} \approx 0.06$

            These events are dependent because drawing the first heart 
            changes the probability of drawing the second heart.
            """,
        },
        "Multiple AND (System Components)": {
            "sets": (0.05, 0.03, 0.0015),  # (CPU fail, disk fail, intersection)
            "labels": ("CPU\nP(C)=0.05", "Disk\nP(D)=0.03"),
            "title": "Independent System Failures",
            "explanation": r"""
            ### System Component Failures

            $P(\text{CPU Failure}) = 0.05$

            $P(\text{Disk Failure}) = 0.03$

            $P(\text{Both Fail}) = 0.05 \times 0.03 = 0.0015$

            Component failures are typically independent in **well-designed systems**,
            meaning one component's failure doesn't affect the other's probability of failing.
            """,
        },
    }

    # Get data for selected event type
    data = events_data[event_type.value]

    # Create visualization
    plt.figure(figsize=(10, 5))
    v = venn2(subsets=data["sets"], set_labels=data["labels"])
    plt.title(data["title"])

    # Display explanation alongside visualization
    mo.hstack([plt.gcf(), mo.md(data["explanation"])])
    return data, events_data, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding

        Which of these statements about AND probability are true?

        <details>
        <summary>1. The probability of getting two sixes in a row with a fair die is 1/36</summary>

        ✅ True! Since die rolls are independent events:
        P(two sixes) = P(first six) × P(second six) = 1/6 × 1/6 = 1/36
        </details>

        <details>
        <summary>2. When drawing cards without replacement, P(two kings) = 4/52 × 4/52</summary>

        ❌ False! This is a dependent event. The correct calculation is:
        P(two kings) = P(first king) × P(second king | first king) = 4/52 × 3/51
        </details>

        <details>
        <summary>3. If P(A) = 0.3 and P(B) = 0.4, then P(A and B) must be 0.12</summary>

        ❌ False! P(A and B) = 0.12 only if A and B are independent events.
        If they're dependent, we need P(B|A) to calculate P(A and B).
        </details>

        <details>
        <summary>4. The probability of rolling a six AND getting tails is (1/6 × 1/2)</summary>

        ✅ True! These are independent events, so we multiply their individual probabilities:
        P(six and tails) = P(six) × P(tails) = 1/6 × 1/2 = 1/12
        </details>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Summary

        You've learned:

        - How to identify independent vs dependent events
        - The multiplication rule for independent events
        - The chain rule for dependent events
        - How to extend these concepts to multiple events

        In the next lesson, we'll explore **law of total probability** in more detail, building on our understanding of various topics.
        """
    )
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Independence in Probability Theory

    _This notebook is a computational companion to the book ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/independence/), by Stanford professor Chris Piech._

    In probability theory, independence is a fundamental concept that helps us understand
    when events don't influence each other. Two events are independent if knowing the
    outcome of one event doesn't change our belief about the other event occurring.

    ## Definition of Independence

    Two events $E$ and $F$ are independent if:

    $$P(E|F) = P(E)$$

    This means that knowing $F$ occurred doesn't change the probability of $E$ occurring.

    ### _Alternative Definition_

    Using the chain rule, we can derive another equivalent definition:

    $$P(E \cap F) = P(E) \cdot P(F)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independence is Symmetric

    This property is symmetric: if $E$ is independent of $F$, then $F$ is independent of $E$.
    We can prove this using Bayes' Theorem:

    \[P(E|F) = \frac{P(F|E)P(E)}{P(F)}\]

    \[= \frac{P(F)P(E)}{P(F)}\]

    \[= P(E)\]

    ## Independence and Complements

    Given independent events $A$ and $B$, we can prove that $A$ and $B^C$ are also independent:


    \[P(AB^C) = P(A) - P(AB)\]

    \[= P(A) - P(A)P(B)\]

    \[= P(A)(1 - P(B))\]

    \[= P(A)P(B^C)\]

    ## Generalized Independence

    Events $E_1, E_2, \ldots, E_n$ are independent if for every subset with $r$ elements (where $r \leq n$):

    \[P(E_1, E_2, \ldots, E_r) = \prod_{i=1}^r P(E_i)\]

    For example, consider getting 5 heads on 5 coin flips. Let $H_i$ be the event that the $i$th flip is heads:


    \[P(H_1, H_2, H_3, H_4, H_5) = P(H_1)P(H_2)P(H_3)P(H_4)P(H_5)\]

    \[= \prod_{i=1}^5 P(H_i)\]

    \[= \left(\frac{1}{2}\right)^5 = 0.03125\]

    ## Conditional Independence

    Events $E_1, E_2, E_3$ are conditionally independent given event $F$ if:

    \[P(E_1, E_2, E_3 | F) = P(E_1|F)P(E_2|F)P(E_3|F)\]

    This can be written more succinctly using product notation:

    \[P(E_1, E_2, E_3 | F) = \prod_{i=1}^3 P(E_i|F)\]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md(r"""While the rules of probability stay the same when conditioning on an event, the independence 
        property between events might change. Events that were dependent can become independent when 
        conditioning on an event. Events that were independent can become dependent.

        For example, if events $E_1, E_2, E_3$ are conditionally independent given event $F$, 
        it is not necessarily true that:

        $$P(E_1,E_2,E_3) = \prod_{i=1}^3 P(E_i)$$

        as we are no longer conditioning on $F$.
        """)
    mo.callout(
        callout_text,
        kind="warn"
    )
    return


@app.function
def check_independence(p_e, p_f, p_intersection):
    expected = p_e * p_f
    tolerance = 1e-5  # Stricter tolerance for comparison

    return abs(p_intersection - expected) < tolerance


@app.cell
def _(mo):
    # Example 1: Rolling dice
    p_first_even = 0.5      # P(First die is even)
    p_second_six = 1/6      # P(Second die is 6)
    p_both = 1/12           # P(First even AND second is 6)
    dice_independent = check_independence(p_first_even, p_second_six, p_both)
    example1 = f"""
    #### 1. Rolling Two Dice 🎲

    - P(First die is even) = {p_first_even:.3f}
    - P(Second die shows 6) = {p_second_six:.3f}
    - P(Both events occur) = {p_both:.3f}

    **Result**: Events are {'independent' if dice_independent else 'dependent'}

    <details>
    <summary>Why?</summary>
    Rolling dice are independent events because the outcome of one die doesn't affect the other. 
    The probability of getting a 6 on the second die (1/6) remains the same regardless of what shows on the first die. This is why P(Both) = 1/12, which equals P(First even) * P(Second is 6) = 0.5 * 1/6.
    </details>
    """
    mo.md(example1)
    return


@app.cell
def _(mo):
    # Example 2: Drawing cards (dependent events)
    p_first_heart = 13/52    # P(First card is heart)
    p_second_heart = 12/51   # P(Second card is heart | First was heart)

    # For dependent events, P(A∩B) = P(A) * P(B|A)
    p_both_hearts = (13/52) * (12/51)  # Joint probability = 0.059

    # If events were independent, we'd expect:
    theoretical_if_independent = (13/52) * (13/52)  # = 0.0625

    # Test independence by comparing actual joint probability with theoretical independent probability
    cards_independent = check_independence(13/52, 13/52, p_both_hearts)

    example2 = f"""
    #### 2. Drawing Cards Without Replacement ♥️
    - P(First card is heart) = {p_first_heart:.3f}
    - P(Second card is heart | First was heart) = {p_second_heart:.3f}
    - P(Both cards are hearts) = {p_both_hearts:.3f}
    - If independent, P(Both hearts) would be = {theoretical_if_independent:.3f}

    **Result**: Events are {'independent' if cards_independent else 'dependent'}

    <details>
    <summary>Why?</summary>
    Drawing cards without replacement makes events dependent. The probability of getting 
    a second heart (12/51 ≈ 0.235) is less than the first (13/52 = 0.25) because there's one fewer heart available. This makes the actual probability of both hearts (0.059) less than what we'd 
    expect if the events were independent (0.063).
    </details>
    """
    mo.md(example2)
    return


@app.cell
def _(mo):
    # Example 3: Computer system
    p_hardware = 0.02       # P(Hardware failure)
    p_software = 0.03       # P(Software crash)
    p_both_failure = 0.0006 # P(Both failures)
    system_independent = check_independence(p_hardware, p_software, p_both_failure)
    example3 = f"""
    #### 3. Computer System Failures 💻

    - P(Hardware failure) = {p_hardware:.3f}
    - P(Software crash) = {p_software:.3f}
    - P(Both failures occur) = {p_both_failure:.3f}

    **Result**: Events are {'independent' if system_independent else 'dependent'}

    <details>
    <summary>Why?</summary>
    Computer hardware and software failures are typically independent because they often have different root causes. Hardware failures might be due to physical issues (heat, power), while software crashes come from programming bugs. This is why P(Both) = 0.0006, which equals P(Hardware) * P(Software) = 0.02 * 0.03.
    </details>
    """
    mo.md(example3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Establishing Independence

    In practice, we can establish independence through:

    1. **Mathematical Verification**: Show that P(E∩F) = P(E)P(F)
    2. **Empirical Testing**: Analyze data to check if events appear independent
    3. **Domain Knowledge**: Use understanding of the system to justify independence

    > **Note**: Perfect independence is rare in real data. We often make independence assumptions
    when dependencies are negligible and the simplification is useful.

    ## Backup Systems in Space Missions

    Consider a space mission with two backup life support systems:

    $$P(	ext{Primary fails}) = p_1$$

    $$P(	ext{Secondary fails}) = p_2$$

    If the systems are truly independent (different power sources, separate locations, distinct technologies):

    $$P(	ext{Life support fails}) = p_1p_2$$

    For example:

    - If $p_1 = 0.01$ and $p_2 = 0.02$ (99% and 98% reliable)
    - Then $P(	ext{Total failure}) = 0.0002$ (99.98% reliable)

    However, if both systems share vulnerabilities (same radiation exposure, temperature extremes):

    $$P(	ext{Life support fails}) > p_1p_2$$

    This example shows why space agencies invest heavily in ensuring true independence of backup systems.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Example
    """)
    return


@app.cell
def _(flip_button, mo, reset_button):
    mo.hstack([flip_button, reset_button], justify='center')
    return


@app.cell(hide_code=True)
def _(mo):
    flip_button = mo.ui.run_button(label="Flip Coins!", kind="info")
    reset_button = mo.ui.run_button(label="Reset", kind="danger")
    stats_display = mo.md("*Click 'Flip Coins!' to start simulation*")
    return flip_button, reset_button


@app.cell(hide_code=True)
def _(flip_button, mo, np, reset_button):
    if reset_button.value or not flip_button.value:
            mo.md("*Click 'Flip Coins!' to start simulation*")
    if flip_button.value:
        coin1 = "H" if np.random.random() < 0.5 else "T"
        coin2 = "H" if np.random.random() < 0.5 else "T"

        # Calculate probabilities for this flip
        p_h1 = 1 if coin1 == "H" else 0
        p_h2 = 1 if coin2 == "H" else 0
        p_both_h = 1 if (coin1 == "H" and coin2 == "H") else 0
        p_product = p_h1 * p_h2

        stats = f"""
        # Current Flip Results

        **Individual Probabilities:**

        - P(Heads on Coin 1) = {p_h1:.3f}
        - P(Heads on Coin 2) = {p_h2:.3f}

        **Testing Independence:**

        - P(Both Heads) = {p_both_h:.3f}
        - P(H₁)P(H₂) = {p_product:.3f}

        **Are they independent?**
        The events appear {'independent' if abs(p_both_h - p_product) < 0.1 else 'dependent'}
        (difference: {abs(p_both_h - p_product):.3f})

        **Current Flip:**

        | Coin 1 | Coin 2 |
        |--------|--------|
        | {coin1} | {coin2} |
        """
    else:
        stats = "*Click 'Flip Coins!' to start simulation*"

    new_stats_display = mo.md(stats)
    new_stats_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Understanding the Simulation

    This simulation demonstrates independence using coin flips, where each coin's outcome is unaffected by the other.

    ### Reading the Results

    1. **Individual Probabilities:**

           - P(H₁): 1 if heads, 0 if tails on first coin
           - P(H₂): 1 if heads, 0 if tails on second coin

    2. **Testing Independence:**

           - P(Both Heads): 1 if both show heads, 0 otherwise
           - P(H₁)P(H₂): Product of individual results

    > **Note**: Each click performs a new independent trial. While a single flip shows binary outcomes (0 or 1),
    the theoretical probability is 0.5 for each coin and 0.25 for both heads.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤔 Test Your Understanding

    Which of these statements about independence are true?

    <details>
    <summary>If P(E|F) = P(E), then E and F are independent</summary>
    ✅ True! This is one definition of independence - knowing F occurred doesn't change the probability of E.
    </details>

    <details>
    <summary>Independent events cannot occur simultaneously</summary>
    ❌ False! Independent events can and do occur together - their joint probability is just the product of their individual probabilities.
    </details>

    <details>
    <summary>If P(E∩F) = P(E)P(F), then E and F are independent</summary>
    ✅ True! This is the multiplicative definition of independence.
    </details>

    <details>
    <summary>Independence is symmetric: if E is independent of F, then F is independent of E</summary>
    ✅ True! The definition P(E∩F) = P(E)P(F) is symmetric in E and F.
    </details>

    <details>
    <summary>Three events being pairwise independent means they are mutually independent</summary>
    ❌ False! Pairwise independence doesn't guarantee mutual independence - we need to check all combinations.
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    In this exploration of probability independence, we've discovered how to recognize when events truly don't influence each other. Through the lens of both mathematical definitions and interactive examples, we've seen how independence manifests in scenarios ranging from simple coin flips to critical system designs.

    The power of independence lies in its simplicity: when events are independent, we can multiply their individual probabilities to understand their joint behavior. Yet, as our examples showed, true independence is often more nuanced than it first appears. What seems independent might harbor hidden dependencies, and what appears dependent might be independent under certain conditions.

    _The art lies not just in calculating probabilities, but in developing the intuition to recognize independence in real-world scenarios—a skill essential for making informed decisions in uncertain situations._
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    return (np,)


if __name__ == "__main__":
    app.run()

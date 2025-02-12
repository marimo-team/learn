# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "matplotlib-venn"
# ]
# ///

import marimo

__generated_with = "0.11.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2
    import numpy as np
    return np, plt, venn2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Probability of Or

        When calculating the probability of either one event _or_ another occurring, we need to be careful about how we combine probabilities. The method depends on whether the events can happen together[<sup>1</sup>](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/prob_or/).

        Let's explore how to calculate $P(E \cup F)$ or $P(E \text{ or } F)$ in different scenarios.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mutually Exclusive Events

        Two events $E$ and $F$ are **mutually exclusive** if they cannot occur simultaneously. 
        In set notation, this means:

        $E \cap F = \emptyset$

        For example:

        - Rolling an even number (2,4,6) vs rolling an odd number (1,3,5)
        - Drawing a heart vs drawing a spade from a deck
        - Passing vs failing a test

        Here's a Python function to check if two sets of outcomes are mutually exclusive:
        """
    )
    return


@app.cell
def _():
    def are_mutually_exclusive(event1, event2):
        return len(event1.intersection(event2)) == 0

    # Example with dice rolls
    even_numbers = {2, 4, 6}
    odd_numbers = {1, 3, 5}
    prime_numbers = {2, 3, 5, 7}
    return are_mutually_exclusive, even_numbers, odd_numbers, prime_numbers


@app.cell
def _(are_mutually_exclusive, even_numbers, odd_numbers):
    are_mutually_exclusive(even_numbers, odd_numbers)
    return


@app.cell
def _(are_mutually_exclusive, even_numbers, prime_numbers):
    are_mutually_exclusive(even_numbers, prime_numbers)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Or with Mutually Exclusive Events

        For mutually exclusive events, the probability of either event occurring is simply the sum of their individual probabilities:

        $P(E \cup F) = P(E) + P(F)$

        This extends to multiple events. For $n$ mutually exclusive events $E_1, E_2, \ldots, E_n$:

        $P(E_1 \cup E_2 \cup \cdots \cup E_n) = \sum_{i=1}^n P(E_i)$

        Let's implement this calculation:
        """
    )
    return


@app.cell
def _():
    def prob_union_mutually_exclusive(probabilities):
        return sum(probabilities)

    # Example: Rolling a die
    # P(even) = P(2) + P(4) + P(6)
    p_even_mutually_exclusive = prob_union_mutually_exclusive([1/6, 1/6, 1/6])
    print(f"P(rolling an even number) = {p_even_mutually_exclusive}")

    # P(prime) = P(2) + P(3) + P(5)
    p_prime_mutually_exclusive = prob_union_mutually_exclusive([1/6, 1/6, 1/6])
    print(f"P(rolling a prime number) = {p_prime_mutually_exclusive}")
    return (
        p_even_mutually_exclusive,
        p_prime_mutually_exclusive,
        prob_union_mutually_exclusive,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Or with Non-Mutually Exclusive Events

        When events can occur together, we need to use the **inclusion-exclusion principle**:

        $P(E \cup F) = P(E) + P(F) - P(E \cap F)$

        Why subtract $P(E \cap F)$? Because when we add $P(E)$ and $P(F)$, we count the overlap twice!

        For example, consider calculating $P(\text{prime or even})$ when rolling a die:
        - Prime numbers: {2, 3, 5}
        - Even numbers: {2, 4, 6}
        - The number 2 is counted twice unless we subtract its probability

        Here's how to implement this calculation:
        """
    )
    return


@app.cell
def _():
    def prob_union_general(p_a, p_b, p_intersection):
        """Calculate probability of union for any two events"""
        return p_a + p_b - p_intersection

    # Example: Rolling a die
    # P(prime or even)
    p_prime_general = 3/6    # P(prime) = P(2,3,5)
    p_even_general = 3/6     # P(even) = P(2,4,6)
    p_intersection = 1/6     # P(intersection) = P(2)

    result = prob_union_general(p_prime_general, p_even_general, p_intersection)
    print(f"P(prime or even) = {p_prime_general} + {p_even_general} - {p_intersection} = {result}")
    return (
        p_even_general,
        p_intersection,
        p_prime_general,
        prob_union_general,
        result,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Extension to Three Events

        For three events, the inclusion-exclusion principle becomes:

        $P(E_1 \cup E_2 \cup E_3) = P(E_1) + P(E_2) + P(E_3)$
        $- P(E_1 \cap E_2) - P(E_1 \cap E_3) - P(E_2 \cap E_3)$
        $+ P(E_1 \cap E_2 \cap E_3)$

        The pattern is:

        1. Add individual probabilities
        2. Subtract probabilities of pairs
        3. Add probability of triple intersection
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Interactive example:""")
    return


@app.cell
def _(event_type):
    event_type
    return


@app.cell(hide_code=True)
def _(mo):
    # Create a dropdown to select the type of events to visualize
    event_type = mo.ui.dropdown(
        options=[
            "Mutually Exclusive Events (Rolling Odd vs Even)",
            "Non-Mutually Exclusive Events (Prime vs Even)",
            "Three Events (Less than 3, Even, Prime)"
        ],
        value="Mutually Exclusive Events (Rolling Odd vs Even)",
        label="Select Event Type"
    )
    return (event_type,)


@app.cell(hide_code=True)
def _(event_type, mo, plt, venn2):
    # Define the events and their probabilities
    events_data = {
        "Mutually Exclusive Events (Rolling Odd vs Even)": {
            "sets": (round(3/6, 2), round(3/6, 2), 0),  # (odd, even, intersection)
            "labels": ("Odd\n{1,3,5}", "Even\n{2,4,6}"),
            "title": "Mutually Exclusive Events: Odd vs Even Numbers",
            "explanation": r"""
            ### Mutually Exclusive Events

            $P(\text{Odd}) = \frac{3}{6} = 0.5$
            $P(\text{Even}) = \frac{3}{6} = 0.5$
            $P(\text{Odd} \cap \text{Even}) = 0$

            $P(\text{Odd} \cup \text{Even}) = P(\text{Odd}) + P(\text{Even}) = 1$

            These events are mutually exclusive because a number cannot be both odd and even.
            """
        },
        "Non-Mutually Exclusive Events (Prime vs Even)": {
            "sets": (round(2/6, 2), round(2/6, 2), round(1/6, 2)),  # (prime-only, even-only, intersection)
            "labels": ("Prime\n{3,5}", "Even\n{4,6}"),
            "title": "Non-Mutually Exclusive: Prime vs Even Numbers",
            "explanation": r"""
            ### Non-Mutually Exclusive Events

            $P(\text{Prime}) = \frac{3}{6} = 0.5$ (2,3,5)
            $P(\text{Even}) = \frac{3}{6} = 0.5$ (2,4,6)
            $P(\text{Prime} \cap \text{Even}) = \frac{1}{6}$ (2)

            $P(\text{Prime} \cup \text{Even}) = \frac{3}{6} + \frac{3}{6} - \frac{1}{6} = \frac{5}{6}$

            These events overlap because 2 is both prime and even.
            """
        },
        "Three Events (Less than 3, Even, Prime)": {
            "sets": (round(1/6, 2), round(2/6, 2), round(1/6, 2)),  # (less than 3, even, intersection)
            "labels": ("<3\n{1,2}", "Even\n{2,4,6}"),
            "title": "Complex Example: Numbers < 3 and Even Numbers",
            "explanation": r"""
            ### Complex Event Interaction

            $P(x < 3) = \frac{2}{6}$ (1,2)
            $P(\text{Even}) = \frac{3}{6}$ (2,4,6)
            $P(x < 3 \cap \text{Even}) = \frac{1}{6}$ (2)

            $P(x < 3 \cup \text{Even}) = \frac{2}{6} + \frac{3}{6} - \frac{1}{6} = \frac{4}{6}$

            The number 2 belongs to both sets, requiring the inclusion-exclusion principle.
            """
        }
    }

    # Get data for selected event type
    data = events_data[event_type.value]

    # Create visualization
    plt.figure(figsize=(10, 5))
    v = venn2(subsets=data["sets"], 
              set_labels=data["labels"])
    plt.title(data["title"])

    # Display explanation alongside visualization
    mo.hstack([
        plt.gcf(),
        mo.md(data["explanation"])
    ])
    return data, events_data, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding

        Consider rolling a six-sided die. Which of these statements are true?

        <details>
        <summary>1. P(even or less than 3) = P(even) + P(less than 3)</summary>

        ❌ Incorrect! These events are not mutually exclusive (2 is both even and less than 3).
        We need to use the inclusion-exclusion principle.
        </details>

        <details>
        <summary>2. P(even or greater than 4) = 4/6</summary>

        ✅ Correct! {2,4,6} ∪ {5,6} = {2,4,5,6}, so probability is 4/6.
        </details>

        <details>
        <summary>3. P(prime or odd) = 5/6</summary>

        ✅ Correct! {2,3,5} ∪ {1,3,5} = {1,2,3,5}, so probability is 5/6.
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

        - How to identify mutually exclusive events
        - The addition rule for mutually exclusive events
        - The inclusion-exclusion principle for overlapping events
        - How to extend these concepts to multiple events

        In the next lesson, we'll explore **conditional probability** - how the probability 
        of one event changes when we know another event has occurred.
        """
    )
    return


if __name__ == "__main__":
    app.run()

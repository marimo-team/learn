# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "matplotlib-venn"
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2
    import numpy as np
    return plt, venn2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Law of Total Probability

    _This notebook is a computational companion to the book ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/law_total/), by Stanford professor Chris Piech._

    The Law of Total Probability is a fundamental rule that helps us calculate probabilities by breaking down complex events into simpler parts. It's particularly useful when we want to compute the probability of an event that can occur through multiple distinct scenarios.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Core Concept

    The Law of Total Probability emerged from a simple but powerful observation: any event E can be broken down into parts based on another event F and its complement Fᶜ.

    ### From Simple Observation to Powerful Law

    Consider an event E that can occur in two ways:

    1. When F occurs (E ∩ F)
    2. When F doesn't occur (E ∩ Fᶜ)

    This leads to our first insight:

    $P(E) = P(E \cap F) + P(E \cap F^c)$

    Applying the chain rule to each term:

    \begin{align}
    P(E) &= P(E \cap F) + P(E \cap F^c) \\
    &= P(E|F)P(F) + P(E|F^c)P(F^c)
    \end{align}

    This two-part version generalizes to any number of [mutually exclusive](marimo.app/https://github.com/marimo-team/learn/blob/main/probability/03_probability_of_or.py) events that cover the sample space:

    $P(A) = \sum_{i=1}^n P(A|B_i)P(B_i)$

    where {B₁, B₂, ..., Bₙ} forms a partition of the sample space.
    """)
    return


@app.cell
def _():
    def is_valid_partition(events, sample_space):
        """Check if events form a valid partition of the sample space"""
        # Check if events are mutually exclusive
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events[i+1:], i+1):
                if event1.intersection(event2):
                    return False

        # Check if events cover sample space
        union = set().union(*events)
        return union == sample_space

    # Example with dice
    sample_space = {1, 2, 3, 4, 5, 6}
    partition1 = [{1, 3, 5}, {2, 4, 6}]  # odd vs even
    partition2 = [{1, 2}, {3, 4}, {5, 6}]  # pairs

    print("Odd/Even partition:", is_valid_partition(partition1, sample_space))
    print("Number pairs partition:", is_valid_partition(partition2, sample_space))
    return (is_valid_partition,)


@app.cell
def _(is_valid_partition):
    # Example: Student Grades
    grade_space = {'A', 'B', 'C', 'D', 'F'}
    passing_partition = [{'A', 'B', 'C'}, {'D', 'F'}]  # Pass/Fail
    letter_groups = [{'A'}, {'B'}, {'C'}, {'D'}, {'F'}]  # Individual grades

    print("Student Grades Examples:")
    print("Pass/Fail partition:", is_valid_partition(passing_partition, grade_space))
    print("Individual grades partition:", is_valid_partition(letter_groups, grade_space))
    return


@app.cell
def _(is_valid_partition):
    # Example: Card Suits
    card_space = {'♠', '♣', '♥', '♦'}
    color_partition = [{'♠', '♣'}, {'♥', '♦'}]  # Black/Red
    invalid_partition = [{'♠', '♥'}, {'♣'}]  # Invalid: Doesn't cover full space

    print("\nPlaying Cards Examples:")
    print("Color-based partition:", is_valid_partition(color_partition, card_space))  # True
    print("Invalid partition:", is_valid_partition(invalid_partition, card_space))    # False
    return


@app.cell(hide_code=True)
def _(mo, plt, venn2):
    # Create Venn diagram for E and F
    plt.figure(figsize=(10, 5))
    v = venn2(subsets=(0.3, 0.4, 0.2), 
              set_labels=('F', 'E'))
    plt.title("Decomposing Event E using F")

    viz_explanation = mo.md(r"""
    ### Visual Intuition

    In this diagram:

    - The red region (E) is split into two parts:
          1. Part inside F (E ∩ F)
          2. Part outside F (E ∩ Fᶜ)

    This visualization shows why:
    $P(E) = P(E|F)P(F) + P(E|F^c)P(F^c)$

    The same principle extends to any number of mutually exclusive parts!
    """)

    mo.hstack([plt.gca(), viz_explanation])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computing Total Probability

    To use the Law of Total Probability:

    1. Identify a partition of the sample space
    2. Calculate $P(B_i)$ for each part
    3. Calculate $P(A|B_i)$ for each part
    4. Sum the products $P(A|B_i)P(B_i)$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's implement this calculation:
    """)
    return


@app.function
def total_probability(conditional_probs, partition_probs):
    """Calculate total probability using Law of Total Probability
    conditional_probs: List of P(A|Bi)
    partition_probs: List of P(Bi)
    """
    if len(conditional_probs) != len(partition_probs):
        raise ValueError("Must have same number of conditional and partition probabilities")

    if abs(sum(partition_probs) - 1) > 1e-10:
        raise ValueError("Partition probabilities must sum to 1")

    return sum(c * p for c, p in zip(conditional_probs, partition_probs))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: System Reliability

    Consider a computer system that can be in three states:

    - Normal (70% of time)
    - Degraded (20% of time)
    - Critical (10% of time)

    The probability of errors in each state:

    - P(Error | Normal) = 0.01 (1%)
    - P(Error | Degraded) = 0.15 (15%)
    - P(Error | Critical) = 0.45 (45%)

    Let's calculate the overall probability of encountering an error:
    """)
    return


@app.cell
def _(mo):
    # System states and probabilities
    states = ["Normal", "Degraded", "Critical"]
    state_probs = [0.7, 0.2, 0.1]  # System spends 70%, 20%, 10% of time in each state
    error_probs = [0.01, 0.15, 0.45]  # Error rates increase with system degradation

    # Calculate total probability
    total_error = total_probability(error_probs, state_probs)

    explanation = mo.md(f"""
    ### System Error Analysis

    Given:

    - Normal State (70% of time):
          - Only 1% chance of errors
    - Degraded State (20% of time):
          - Higher 15% chance of errors
    - Critical State (10% of time):
          - Highest 45% chance of errors

    Using Law of Total Probability:
    $P(\text{{Error}}) = \sum_{{i=1}}^3 P(\text{{Error}}|B_i)P(B_i)$

    Step by step:

    1. Normal: 0.01 × 0.7 = 0.007 (0.7%)
    2. Degraded: 0.15 × 0.2 = 0.030 (3.0%)
    3. Critical: 0.45 × 0.1 = 0.045 (4.5%)

    Total: {total_error:.3f} or {total_error:.1%} chance of error
    """)
    explanation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Example:
    """)
    return


@app.cell
def _(late_given_dry, late_given_rain, mo, weather_prob):
    mo.hstack([weather_prob, late_given_rain, late_given_dry])
    return


@app.cell(hide_code=True)
def _(mo):
    # Create sliders for interactive example
    weather_prob = mo.ui.slider(0, 1, value=0.3, label="P(Rain)")
    late_given_rain = mo.ui.slider(0, 1, value=0.6, label="P(Late|Rain)")
    late_given_dry = mo.ui.slider(0, 1, value=0.2, label="P(Late|No Rain)")
    return late_given_dry, late_given_rain, weather_prob


@app.cell
def _(late_given_dry, late_given_rain, mo, plt, venn2, weather_prob):
    # Calculate probabilities
    p_rain = weather_prob.value
    p_dry = 1 - p_rain
    p_late = late_given_rain.value * p_rain + late_given_dry.value * p_dry

    # Create explanation
    explanation_example = mo.md(f"""
    ### Weather and Traffic Analysis

    Given:

    - P(Rain) = {p_rain:.2f}
    - P(No Rain) = {p_dry:.2f}
    - P(Late|Rain) = {late_given_rain.value:.2f}
    - P(Late|No Rain) = {late_given_dry.value:.2f}

    Using Law of Total Probability:

    $P(\text{{Late}}) = P(\text{{Late}}|\text{{Rain}})P(\text{{Rain}}) + P(\text{{Late}}|\text{{No Rain}})P(\text{{No Rain}})$

    $P(\text{{Late}}) = ({late_given_rain.value:.2f} \ times {p_rain:.2f}) + ({late_given_dry.value:.2f} \ times {p_dry:.2f}) = {p_late:.2f}$
    """)

    # Visualize with Venn diagram
    plt.figure(figsize=(10, 5))
    _v = venn2(subsets=(
        round(p_rain * (1 - late_given_rain.value), 2),  # Rain only
        round(p_dry * (1 - late_given_dry.value), 2),    # No Rain only
        round(p_rain * late_given_rain.value, 2)         # Intersection
    ), set_labels=('Rain', 'Late'))
    plt.title("Weather and Traffic Probability")

    mo.hstack([plt.gca(), explanation_example])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visual Intuition

    The Law of Total Probability works because:

    1. The partition divides the sample space into non-overlapping regions
    2. Every outcome belongs to exactly one region
    3. We account for all possible ways an event can occur

    Let's visualize this with a tree diagram:
    """)
    return


@app.cell(hide_code=True)
def _(plt):
    # Create tree diagram with better spacing
    plt.figure(figsize=(12, 8))

    # First level - partition probabilities sum to 1
    plt.plot([0, 2], [6, 9], 'k-', linewidth=2)  # B₁ branch
    plt.plot([0, 2], [6, 6], 'k-', linewidth=2)  # B₂ branch
    plt.plot([0, 2], [6, 3], 'k-', linewidth=2)  # B₃ branch

    # Second level - conditional probabilities sum to 1 for each branch
    plt.plot([2, 4], [9, 10], 'b-', linewidth=2)  # A|B₁
    plt.plot([2, 4], [9, 8], 'r-', linewidth=2)   # Aᶜ|B₁
    plt.plot([2, 4], [6, 7], 'b-', linewidth=2)   # A|B₂
    plt.plot([2, 4], [6, 5], 'r-', linewidth=2)   # Aᶜ|B₂
    plt.plot([2, 4], [3, 4], 'b-', linewidth=2)   # A|B₃
    plt.plot([2, 4], [3, 2], 'r-', linewidth=2)   # Aᶜ|B₃

    # Add labels with actual probabilities
    plt.text(0, 6.2, 'S (1.0)', fontsize=12)
    plt.text(2, 9.2, 'B₁ (1/3)', fontsize=12)
    plt.text(2, 6.2, 'B₂ (1/3)', fontsize=12)
    plt.text(2, 3.2, 'B₃ (1/3)', fontsize=12)

    # Add conditional probability labels
    plt.text(4, 10.2, 'A (P(A|B₁))', fontsize=10, color='blue')
    plt.text(4, 7.8, 'Aᶜ (1-P(A|B₁))', fontsize=10, color='red')
    plt.text(4, 7.2, 'A (P(A|B₂))', fontsize=10, color='blue')
    plt.text(4, 4.8, 'Aᶜ (1-P(A|B₂))', fontsize=10, color='red')
    plt.text(4, 4.2, 'A (P(A|B₃))', fontsize=10, color='blue')
    plt.text(4, 1.8, 'Aᶜ (1-P(A|B₃))', fontsize=10, color='red')

    plt.axis('off')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤔 Test Your Understanding

    For a fair six-sided die with partitions:
    - B₁: Numbers less than 3 {1,2}
    - B₂: Numbers from 3 to 4 {3,4}
    - B₃: Numbers greater than 4 {5,6}

    **Question 1**: Which of these statements correctly describes the partition?
    <details>
    <summary>The sets overlap at number 3</summary>
    ❌ Incorrect! The sets are clearly separated with no overlapping numbers.
    </details>
    <details>
    <summary>Some numbers are missing from the partition</summary>
    ❌ Incorrect! All numbers from 1 to 6 are included exactly once.
    </details>
    <details>
    <summary>The sets form a valid partition of {1,2,3,4,5,6}</summary>
    ✅ Correct! The sets are mutually exclusive and their union covers all outcomes.
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    You've learned:

    - How to identify valid partitions of a sample space
    - The Law of Total Probability formula and its components
    - How to break down complex probability calculations
    - Applications to real-world scenarios

    In the next lesson, we'll explore **Bayes' Theorem**, which builds on these concepts to solve even more sophisticated probability problems.
    """)
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "matplotlib-venn==1.1.1",
#     "numpy==2.2.2",
# ]
# ///

import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium", app_title="Conditional Probability")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3
    import numpy as np
    return np, plt, venn3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Conditional Probability

        In probability theory, we often want to update our beliefs when we receive new information. 
        Conditional probability helps us formalize this process by calculating "_what is the chance of 
        event $E$ happening given that we have already observed some other event $F$?_"[<sup>1</sup>](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/cond_prob/)

        When we condition on an event $F$:

        - We enter the universe where $F$ has occurred
        - Only outcomes consistent with $F$ are possible
        - Our sample space reduces to $F$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Definition of Conditional Probability

        The probability of event $E$ given that event $F$ has occurred is denoted as $P(E|F)$ and is defined as:

        $$P(E|F) = \frac{P(E \cap F)}{P(F)}$$

        This formula tells us that the conditional probability is the probability of both events occurring 
        divided by the probability of the conditioning event.

        Let's understand this with a function that computes conditional probability:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, plt, venn3):
    # Create figure with square boundaries
    plt.figure(figsize=(10, 5))

    # Draw square sample space first
    rect = plt.Rectangle((-2, -2), 4, 4, fill=False, color='gray', linestyle='--')
    plt.gca().add_patch(rect)

    # Set the axis limits to show the full rectangle
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    # Create Venn diagram showing E and F
    # For venn3, subsets order is: (100, 010, 110, 001, 101, 011, 111)
    # Representing: (A, B, AB, C, AC, BC, ABC)
    v = venn3(subsets=(30, 20, 10, 40, 0, 0, 0),
             set_labels=('E', 'F', 'Rest'))

    # Customize colors
    if v:
        for id in ['100', '010', '110', '001']:
            if v.get_patch_by_id(id):
                if id == '100':
                    v.get_patch_by_id(id).set_color('#ffcccc')  # Light red for E
                elif id == '010':
                    v.get_patch_by_id(id).set_color('#ccffcc')  # Light green for F
                elif id == '110':
                    v.get_patch_by_id(id).set_color('#e6ffe6')  # Lighter green for intersection
                elif id == '001':
                    v.get_patch_by_id(id).set_color('white')    # White for rest

    plt.title('Conditional Probability in Sample Space')

    # Remove ticks but keep the box visible
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.axis('on')

    # Add sample space annotation with arrow
    plt.annotate('Sample Space (100)', 
                xy=(-1.5, 1.5),
                xytext=(-2.2, 2),
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray'),
                arrowprops=dict(arrowstyle='->'))

    # Add explanation
    explanation = mo.md(r"""
    ### Visual Intuition

    In our sample space of 100 outcomes:

    - Event $E$ occurs in 40 cases (red region: 30 + 10)
    - Event $F$ occurs in 30 cases (green region: 20 + 10)
    - Both events occur together in 10 cases (overlap)
    - Remaining cases: 40 (to complete sample space of 100)

    When we condition on $F$:
    $$P(E|F) = \frac{P(E \cap F)}{P(F)} = \frac{10}{30} = \frac{1}{3} \approx 0.33$$

    This means: When we know $F$ has occurred (restricting ourselves to the green region),
    the probability of $E$ also occurring is $\frac{1}{3}$ - as 10 out of the 30 cases in the 
    green region also belong to the red region.
    """)

    mo.hstack([plt.gcf(), explanation])
    return explanation, id, rect, v


@app.cell
def _():
    def conditional_probability(p_intersection, p_condition):
        if p_condition == 0:
            raise ValueError("Cannot condition on an impossible event")
        if p_intersection > p_condition:
            raise ValueError("P(E∩F) cannot be greater than P(F)")

        return p_intersection / p_condition
    return (conditional_probability,)


@app.cell
def _(conditional_probability):
    # Example 1: Rolling a die
    # E: Rolling an even number (2,4,6)
    # F: Rolling a number greater than 3 (4,5,6)
    p_even_given_greater_than_3 = conditional_probability(2/6, 3/6)
    print("Example 1: Rolling a die")
    print(f"P(Even | >3) = {p_even_given_greater_than_3}")  # Should be 2/3
    return (p_even_given_greater_than_3,)


@app.cell
def _(conditional_probability):
    # Example 2: Cards
    # E: Drawing a Heart
    # F: Drawing a Face card (J,Q,K)
    p_heart_given_face = conditional_probability(3/52, 12/52)
    print("\nExample 2: Drawing cards")
    print(f"P(Heart | Face card) = {p_heart_given_face}")  # Should be 1/4
    return (p_heart_given_face,)


@app.cell
def _(conditional_probability):
    # Example 3: Student grades
    # E: Getting an A
    # F: Studying more than 3 hours
    p_a_given_study = conditional_probability(0.24, 0.40)
    print("\nExample 3: Student grades")
    print(f"P(A | Studied >3hrs) = {p_a_given_study}")  # Should be 0.6
    return (p_a_given_study,)


@app.cell
def _(conditional_probability):
    # Example 4: Weather
    # E: Raining
    # F: Cloudy
    p_rain_given_cloudy = conditional_probability(0.15, 0.30)
    print("\nExample 4: Weather")
    print(f"P(Rain | Cloudy) = {p_rain_given_cloudy}")  # Should be 0.5
    return (p_rain_given_cloudy,)


@app.cell
def _(conditional_probability):
    # Example 5: Error cases
    print("\nExample 5: Error cases")
    try:
        # Cannot condition on impossible event
        conditional_probability(0.5, 0)
    except ValueError as e:
        print(f"Error 1: {e}")

    try:
        # Intersection cannot be larger than condition
        conditional_probability(0.7, 0.5)
    except ValueError as e:
        print(f"Error 2: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Conditional Paradigm

        When we condition on an event, we enter a new probability universe. In this universe:

        1. All probability axioms still hold
        2. We must consistently condition on the same event
        3. Our sample space becomes the conditioning event

        Here's how our familiar probability rules look when conditioned on event $G$:

        | Rule | Original | Conditioned on $G$ |
        |------|----------|-------------------|
        | Axiom 1 | $0 \leq P(E) \leq 1$ | $0 \leq P(E\|G) \leq 1$ |
        | Axiom 2 | $P(S) = 1$ | $P(S\|G) = 1$ |
        | Axiom 3* | $P(E \cup F) = P(E) + P(F)$ | $P(E \cup F\|G) = P(E\|G) + P(F\|G)$ |
        | Complement | $P(E^C) = 1 - P(E)$ | $P(E^C\|G) = 1 - P(E\|G)$ |

        *_For mutually exclusive events_
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Multiple Conditions

        We can condition on multiple events. The notation $P(E|F,G)$ means "_the probability of $E$ 
        occurring, given that both $F$ and $G$ have occurred._"

        The conditional probability formula still holds in the universe where $G$ has occurred:

        $$P(E|F,G) = \frac{P(E \cap F|G)}{P(F|G)}$$

        This is a powerful extension that allows us to update our probabilities as we receive 
        multiple pieces of information.
        """
    )
    return


@app.cell
def _():
    def multiple_conditional_probability(p_intersection_all, p_intersection_conditions, p_condition):
        """Calculate P(E|F,G) = P(E∩F|G)/P(F|G) = P(E∩F∩G)/P(F∩G)"""
        if p_condition == 0:
            raise ValueError("Cannot condition on an impossible event")
        if p_intersection_conditions == 0:
            raise ValueError("Cannot condition on an impossible combination of events")
        if p_intersection_all > p_intersection_conditions:
            raise ValueError("P(E∩F∩G) cannot be greater than P(F∩G)")

        return p_intersection_all / p_intersection_conditions
    return (multiple_conditional_probability,)


@app.cell
def _(multiple_conditional_probability):
    # Example: College admissions
    # E: Getting admitted
    # F: High GPA
    # G: Good test scores

    # P(E∩F∩G) = P(Admitted ∩ HighGPA ∩ GoodScore) = 0.15
    # P(F∩G) = P(HighGPA ∩ GoodScore) = 0.25

    p_admit_given_both = multiple_conditional_probability(0.15, 0.25, 0.25)
    print("College Admissions Example:")
    print(f"P(Admitted | High GPA, Good Scores) = {p_admit_given_both}")  # Should be 0.6

    # Error case: impossible condition
    try:
        multiple_conditional_probability(0.3, 0.2, 0.2)
    except ValueError as e:
        print(f"\nError case: {e}")
    return (p_admit_given_both,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding

        Which of these statements about conditional probability are true?

        <details>
        <summary>Knowing F occurred always decreases the probability of E</summary>
        ❌ False! Conditioning on F can either increase or decrease P(E), depending on how E and F are related.
        </details>

        <details>
        <summary>P(E|F) represents entering a new probability universe where F has occurred</summary>
        ✅ True! We restrict ourselves to only the outcomes where F occurred, making F our new sample space.
        </details>

        <details>
        <summary>If P(E|F) = P(E), then E and F must be the same event</summary>
        ❌ False! This actually means E and F are independent - knowing one doesn't affect the other.
        </details>

        <details>
        <summary>P(E|F) can be calculated by dividing P(E∩F) by P(F)</summary>
        ✅ True! This is the fundamental definition of conditional probability.
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

        - How conditional probability updates our beliefs with new information
        - The formula $P(E|F) = P(E \cap F)/P(F)$ and its intuition
        - How probability rules work in conditional universes
        - How to handle multiple conditions

        In the next lesson, we'll explore **independence** - when knowing about one event 
        tells us nothing about another.
        """
    )
    return


if __name__ == "__main__":
    app.run()

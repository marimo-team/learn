# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.3",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Bayes Theorem")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayes' Theorem

    _This notebook is a computational companion to the book ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/bayes_theorem/), by Stanford professor Chris Piech._

    In the 1740s, an English minister named Thomas Bayes discovered a profound mathematical relationship that would revolutionize how we reason about uncertainty. His theorem provides an elegant framework for calculating the probability of a hypothesis being true given observed evidence.

    At its core, Bayes' Theorem connects two different types of probabilities: the probability of a hypothesis given evidence $P(H|E)$, and its reverse - the probability of evidence given a hypothesis $P(E|H)$. This relationship is particularly powerful because it allows us to compute difficult probabilities using ones that are easier to measure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Heart of Bayesian Reasoning

    The fundamental insight of Bayes' Theorem lies in its ability to relate what we want to know with what we can measure. When we observe evidence $E$, we often want to know the probability of a hypothesis $H$ being true. However, it's typically much easier to measure how likely we are to observe the evidence when we know the hypothesis is true.

    This reversal of perspective - from $P(H|E)$ to $P(E|H)$ - is powerful because it lets us:
    1. Start with what we know (prior beliefs)
    2. Use easily measurable relationships (likelihood)
    3. Update our beliefs with new evidence

    This approach mirrors both how humans naturally learn and the scientific method: we begin with prior beliefs, gather evidence, and update our understanding based on that evidence. This makes Bayes' Theorem not just a mathematical tool, but a framework for rational thinking.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Formula

    Bayes' Theorem states:

    $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$

    Where:

    - $P(H|E)$ is the **posterior probability** - probability of hypothesis H given evidence E
    - $P(E|H)$ is the **likelihood** - probability of evidence E given hypothesis H
    - $P(H)$ is the **prior probability** - initial probability of hypothesis H
    - $P(E)$ is the **evidence** - total probability of observing evidence E

    The denominator $P(E)$ can be expanded using the [Law of Total Probability](https://marimo.app/gh/marimo-team/learn/main?entrypoint=probability%2F07_law_of_total_probability.py):

    $P(E) = P(E|H)P(H) + P(E|H^c)P(H^c)$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding Each Component

    ### 1. Prior Probability - $P(H)$
    - Initial belief about hypothesis before seeing evidence
    - Based on previous knowledge or assumptions
    - Example: Probability of having a disease before any tests

    ### 2. Likelihood - $P(E|H)$
    - Probability of evidence given hypothesis is true
    - Often known from data or scientific studies
    - Example: Probability of positive test given disease present

    ### 3. Evidence - $P(E)$
    - Total probability of observing the evidence
    - Acts as a normalizing constant
    - Can be calculated using Law of Total Probability

    ### 4. Posterior - $P(H|E)$
    - Updated probability after considering evidence
    - Combines prior knowledge with new evidence
    - Becomes new prior for future updates
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Real-World Examples

    ### 1. Medical Testing
    - **Want to know**: $P(\text{Disease}|\text{Positive})$ - Probability of disease given positive test
    - **Easy to know**: $P(\text{Positive}|\text{Disease})$ - Test accuracy for sick people
    - **Causality**: Disease causes test results, not vice versa

    ### 2. Student Ability
    - **Want to know**: $P(\text{High Ability}|\text{Good Grade})$ - Probability student is skilled given good grade
    - **Easy to know**: $P(\text{Good Grade}|\text{High Ability})$ - Probability good students get good grades
    - **Causality**: Ability influences grades, not vice versa

    ### 3. Cell Phone Location
    - **Want to know**: $P(\text{Location}|\text{Signal Strength})$ - Probability of phone location given signal
    - **Easy to know**: $P(\text{Signal Strength}|\text{Location})$ - Signal strength at known locations
    - **Causality**: Location determines signal strength, not vice versa

    These examples highlight a common pattern: what we want to know (posterior) is harder to measure directly than its reverse (likelihood).
    """)
    return


@app.function
def calculate_posterior(prior, likelihood, false_positive_rate):
    # Calculate P(E) using Law of Total Probability
    p_e = likelihood * prior + false_positive_rate * (1 - prior)

    # Calculate posterior using Bayes' Theorem
    posterior = (likelihood * prior) / p_e
    return posterior, p_e


@app.cell
def _():
    # Medical test example
    p_disease = 0.01  # Prior: 1% have the disease
    p_positive_given_disease = 0.95  # Likelihood: 95% test accuracy
    p_positive_given_healthy = 0.10  # False positive rate: 10%

    medical_posterior, medical_evidence = calculate_posterior(
        p_disease,
        p_positive_given_disease,
        p_positive_given_healthy
    )
    return (medical_posterior,)


@app.cell
def _(medical_explanation):
    medical_explanation
    return


@app.cell(hide_code=True)
def _(medical_posterior, mo):
    medical_explanation = mo.md(f"""
    ### Medical Testing Example

    Consider a medical test for a rare disease:

    - Prior: 1% of population has the disease
    - Likelihood: 95% test accuracy for sick people
    - False positive: 10% of healthy people test positive

    Using Bayes' Theorem:
    $P(D|+) = \\frac{{0.95 times 0.01}}{{0.95 times 0.01 + 0.10 times 0.99}} = {medical_posterior:.3f}$

    Despite a positive test, there's only a {medical_posterior:.1%} chance of having the disease!
    This counterintuitive result occurs because the disease is rare (low prior probability).
    """)
    return (medical_explanation,)


@app.cell
def _():
    # Student ability example
    p_high_ability = 0.30  # Prior: 30% of students have high ability
    p_good_grade_given_high = 0.90  # Likelihood: 90% of high ability students get good grades
    p_good_grade_given_low = 0.40  # 40% of lower ability students also get good grades

    student_posterior, student_evidence = calculate_posterior(
        p_high_ability,
        p_good_grade_given_high,
        p_good_grade_given_low
    )
    return (student_posterior,)


@app.cell
def _(student_explanation):
    student_explanation
    return


@app.cell(hide_code=True)
def _(mo, student_posterior):
    student_explanation = mo.md(f"""
    ### Student Ability Example

    If a student gets a good grade, what's the probability they have high ability?

    Using Bayes' Theorem:

    - Prior: 30% have high ability
    - Likelihood: 90% of high ability students get good grades
    - False positive: 40% of lower ability students get good grades

    Result: P(High Ability|Good Grade) = {student_posterior:.2f}

    So a good grade increases our confidence in high ability from 30% to {student_posterior:.1%}
    """)
    return (student_explanation,)


@app.cell
def _():
    # Cell phone location example
    p_location_a = 0.25  # Prior probability of being in location A
    p_strong_signal_at_a = 0.85  # Likelihood of strong signal at A
    p_strong_signal_elsewhere = 0.15  # False positive rate

    location_posterior, location_evidence = calculate_posterior(
        p_location_a,
        p_strong_signal_at_a,
        p_strong_signal_elsewhere
    )
    return (location_posterior,)


@app.cell
def _(location_explanation):
    location_explanation
    return


@app.cell(hide_code=True)
def _(location_posterior, mo):
    location_explanation = mo.md(f"""
    ### Cell Phone Location Example

    Given a strong signal, what's the probability the phone is in location A?

    Using Bayes' Theorem:

    - Prior: 25% chance of being in location A
    - Likelihood: 85% chance of strong signal at A
    - False positive: 15% chance of strong signal elsewhere

    Result: P(Location A|Strong Signal) = {location_posterior:.2f}

    The strong signal increases our confidence in location A from 25% to {location_posterior:.1%}
    """)
    return (location_explanation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive example
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        _This interactive example was made with [marimo](https://github.com/marimo-team/marimo/blob/main/examples/misc/bayes_theorem.py), and is [based on an explanation of Bayes' Theorem by Grant Sanderson](https://www.youtube.com/watch?v=HZGCoVF3YvM&list=PLzq7odmtfKQw2KIbQq0rzWrqgifHKkPG1&index=1&t=3s)_.

        Bayes theorem provides a convenient way to calculate the probability
        of a hypothesis event $H$ given evidence $E$:

        \[
        P(H \mid E) = \frac{P(H) P(E \mid H)}{P(E)}.
        \]


        **The numerator.** The numerator is the probability of events $E$ and $H$ happening
        together; that is,

        \[
           P(H) P(E \mid H) = P(E \cap H).
        \]

        **The denominator.**
        In most calculations, it is helpful to rewrite the denominator $P(E)$ as 

        \[
        P(E) = P(H)P(E \mid H) + P(\neg H) P (E \mid \neg H),
        \]

        which in turn can also be written as


        \[
        P(E) = P(E \cap H) + P(E \cap \neg H).
        \]
        """
    ).left()
    return


@app.cell(hide_code=True)
def _(
    bayes_result,
    construct_probability_plot,
    mo,
    p_e,
    p_e_given_h,
    p_e_given_not_h,
    p_h,
):
    mo.hstack(
        [
            mo.md(
                rf"""
                ### Probability parameters

                You can configure the probabilities of the events $H$, $E \mid H$, and $E \mid \neg H$

                {mo.as_html([p_h, p_e_given_h, p_e_given_not_h])}

                The plot on the right visualizes the probabilities of these events. 

                1. The yellow rectangle represents the event $H$, and its area is $P(H) = {p_h.value:0.2f}$.
                2. The teal rectangle overlapping with the yellow one represents the event $E \cap H$, and
                   its area is $P(H) \cdot P(E \mid H) = {p_h.value * p_e_given_h.value:0.2f}$.
                3. The teal rectangle that doesn't overlap the yellow rectangle represents the event $E \cap \neg H$, and
                   its area is $P(\neg H) \cdot P(E \mid \neg H) = {(1 - p_h.value) * p_e_given_not_h.value:0.2f}$.

                Notice that the sum of the areas in $2$ and $3$ is the probability $P(E) = {p_e:0.2f}$. 

                One way to think about Bayes' Theorem is the following: the probability $P(H \mid E)$ is the probability
                of $E$ and $H$ happening together (the area of the rectangle $2$), divided by the probability of $E$ happening
                at all (the sum of the areas of $2$ and $3$).
                In this case, Bayes' Theorem says

                \[
                P(H \mid E) = \frac{{P(H) P(E \mid H)}}{{P(E)}} = \frac{{{p_h.value} \cdot {p_e_given_h.value}}}{{{p_e:0.2f}}} = {bayes_result:0.2f}
                \]
                """
            ),
            construct_probability_plot(),
        ],
        justify="start",
        gap=4,
        align="start",
        widths=[0.33, 0.5],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Applications in Computer Science

    Bayes' Theorem is fundamental in many computing applications:

    1. **Spam Filtering**

        - $P(\text{Spam}|\text{Words})$ = Probability email is spam given its words
        - Updates as new emails are classified

    2. **Machine Learning**

        - Naive Bayes classifiers
        - Probabilistic graphical models
        - Bayesian neural networks

    3. **Computer Vision**

        - Object detection confidence
        - Face recognition systems
        - Image classification
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🤔 Test Your Understanding

    Pick which of these statements about Bayes' Theorem you think are correct:

    <details>
    <summary>The posterior probability will always be larger than the prior probability</summary>
    ❌ Incorrect! Evidence can either increase or decrease our belief in the hypothesis. For example, a negative medical test decreases the probability of having a disease.
    </details>

    <details>
    <summary>If the likelihood is 0.9 and the prior is 0.5, then the posterior must equal 0.9</summary>
    ❌ Incorrect! We also need the false positive rate to calculate the posterior probability. The likelihood alone doesn't determine the posterior.
    </details>

    <details>
    <summary>The denominator acts as a normalizing constant to ensure the posterior is a valid probability</summary>
    ✅ Correct! The denominator ensures the posterior probability is between 0 and 1 by considering all ways the evidence could occur.
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    You've learned:

    - The components and intuition behind Bayes' Theorem
    - How to update probabilities when new evidence arrives
    - Why posterior probabilities can be counterintuitive
    - Real-world applications in computer science

    In the next lesson, we'll explore Random Variables, which help us work with numerical outcomes in probability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Appendix
    Below (hidden) cell blocks are responsible for the interactive example above
    """)
    return


@app.cell(hide_code=True)
def _(p_e_given_h, p_e_given_not_h, p_h):
    p_e = p_h.value*p_e_given_h.value + (1 - p_h.value)*p_e_given_not_h.value
    bayes_result = p_h.value * p_e_given_h.value / p_e
    return bayes_result, p_e


@app.cell(hide_code=True)
def _(mo):
    p_h = mo.ui.slider(0.0, 1, label="$P(H)$", value=0.1, step=0.1)
    p_e_given_h = mo.ui.slider(0.0, 1, label="$P(E \mid H)$", value=0.3, step=0.1)
    p_e_given_not_h = mo.ui.slider(
        0.0, 1, label=r"$P(E \mid \neg H)$", value=0.3, step=0.1
    )
    return p_e_given_h, p_e_given_not_h, p_h


@app.cell(hide_code=True)
def _(p_e_given_h, p_e_given_not_h, p_h):
    def construct_probability_plot():
        import matplotlib.pyplot as plt

        plt.axes()

        # Radius: 1, face-color: red, edge-color: blue
        plt.figure(figsize=(6,6))
        base = plt.Rectangle((0, 0), 1, 1, fc="black", ec="white", alpha=0.25)
        h = plt.Rectangle((0, 0), p_h.value, 1, fc="yellow", ec="white", label="H")
        e_given_h = plt.Rectangle(
            (0, 0),
            p_h.value,
            p_e_given_h.value,
            fc="teal",
            ec="white",
            alpha=0.5,
            label="E",
        )
        e_given_not_h = plt.Rectangle(
            (p_h.value, 0), 1 - p_h.value, p_e_given_not_h.value, fc="teal", ec="white", alpha=0.5
        )
        plt.gca().add_patch(base)
        plt.gca().add_patch(h)
        plt.gca().add_patch(e_given_not_h)
        plt.gca().add_patch(e_given_h)
        plt.legend()
        return plt.gca()
    return (construct_probability_plot,)


if __name__ == "__main__":
    app.run()

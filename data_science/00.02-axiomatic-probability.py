# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.19"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    title = mo.md("# 🎲 The Rules of Probability: A Beginner's Guide")
    subtitle = mo.md("""
    Think of axioms as the 'rules of the game' 
    that all probabilities must follow. We'll explore these rules using simple examples.
    """)

    mo.hstack([
        mo.vstack([title, subtitle]),
        mo.image("https://w7.pngwing.com/pngs/774/967/png-transparent-two-white-dice-illustration-black-white-dice-bunco-dice-s-free-game-angle-black-white-thumbnail.png", width=150)
    ])
    return subtitle, title


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## The Three Fundamental Axioms

        Probability theory is built on three fundamental axioms:

        1. **Non-Negativity**: Probabilities can't be negative
           - $P(A) ≥ 0$ for any event A

        2. **Normalization**: The probability of the entire sample space is 1
           - $P(S) = 1$ where S is the sample space

        3. **Additivity**: For mutually exclusive events, probabilities add
           - $P(A \cup B) = P(A) + P(B)$ when $A \cap B = \emptyset$

        Let's explore these with simple examples.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Understanding Axioms Through Events

        Before we explore each axiom in detail, let's use a simple dice roll to see how these rules work in practice. 
        This interactive example demonstrates all three axioms:

        - All probabilities shown will be 0 or greater
        - The probability of all possible rolls (1-6) is 1
        - When we consider events like "rolling an even number", we're adding individual probabilities

        Try different events below to see how probabilities behave according to these rules:
        """
    )
    return


@app.cell
def _(event):
    event
    return


@app.cell
def _(mo):
    # Single interactive example with dice
    event = mo.ui.dropdown(
        options=[
            "Rolling a number from 1 to 6",
            "Rolling a 7",
            "Rolling an even number",
            "Rolling a negative number"
        ],
        value="Rolling a number from 1 to 6",
        label="Choose an Event"
    )
    return (event,)


@app.cell
def _(event, mo):
    probability_map = {
        "Rolling a number from 1 to 6": 1.0,
        "Rolling a 7": 0.0,
        "Rolling an even number": 0.5,
        "Rolling a negative number": 0.0
    }

    prob = probability_map[event.value]

    explanation = {
        "Rolling a number from 1 to 6": "Certain event (must happen)",
        "Rolling a 7": "Impossible event (can't happen)",
        "Rolling an even number": "Possible event (might happen)",
        "Rolling a negative number": "Impossible event (can't happen)"
    }

    mo.hstack([
        mo.md(f"""
        ### Event Analysis

        **Probability**: {prob}

        **Why?** {explanation[event.value]}
        """),
        mo.md(f"### Visual Scale\n{'🟦' * int(prob * 10)}{'⬜' * int((1-prob) * 10)}")
    ])
    return explanation, prob, probability_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## First Axiom: Non-Negativity

        The first axiom states that probabilities can't be negative.

        $P(A) ≥ 0$ for any event A

        This makes intuitive sense - you can't have a negative chance of something happening!

        **Example**: When rolling a die

        - P(rolling a 6) = 1/6 ≈ 0.167 (positive)

        - P(rolling a negative number) = 0 (zero, but not negative)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Second Axiom: Total Probability

        The second axiom states that the probability of all possible outcomes together must equal 1.

        $P(S) = 1$ where S is the sample space (all possible outcomes)

        **Example**: Rolling a number from 1 to 6

        - P(rolling a 1) = 1/6
        - P(rolling a 2) = 1/6
        - P(rolling a 3) = 1/6
        - P(rolling a 4) = 1/6
        - P(rolling a 5) = 1/6
        - P(rolling a 6) = 1/6

        Total: 1/6 + 1/6 + 1/6 + 1/6 + 1/6 + 1/6 = 1

        This satisfies our axiom because all possible outcomes sum to 1.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Third Axiom: Additivity

        For events that cannot occur simultaneously (mutually exclusive), 
        their probabilities add.

        **Example**: Rolling a die

        - P(rolling a 1 or 2) = P(rolling a 1) + P(rolling a 2)

        - 2/6 = 1/6 + 1/6

        This works because you can't roll a 1 and 2 simultaneously.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    key_points = mo.md("""
    ## 🎯 Key Takeaways

    1. Probabilities are always between 0 and 1
    2. Impossible events have probability 0
    3. Certain events have probability 1
    4. All probabilities in a sample space sum to 1
    """)

    next_topics = mo.md("""
    ## 📚 Coming Up Next

    - Addition Laws of Probability
        - Simple Events
        - Mutually Exclusive Events
        - General Addition Rule

    Moving towards Core Probability Laws! 🚀
    """)

    mo.hstack([
        mo.callout(key_points, kind="info"),
        mo.callout(next_topics, kind="success")
    ])
    return key_points, next_topics


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    title = mo.md("# 🎲 The Rules of Probability: A Beginner's Guide")
    subtitle = mo.md("""
    Welcome to the world of probability axioms! Think of axioms as the 'rules of the game' 
    that all probabilities must follow. We'll explore these rules using simple, everyday examples.
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
        ## 🌟 The Three Basic Rules

        Before we dive into formal axioms, let's understand three simple rules about probability:

        1. Probabilities are always between 0 and 1
        2. The probability of a certain event is 1
        3. The probability of impossible events is 0

        Let's explore these with a simple dice roll! 🎲
        """
    )
    return


@app.cell
def _(event):
    event
    return


@app.cell
def _(mo):
    # dice probability explorer
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
        ## 🎯 First Axiom: Non-Negativity

        The first axiom states that probabilities can't be negative.

        $P(A) ≥ 0$ for any event A

        This makes intuitive sense - you can't have a negative chance of something happening!
        """
    )
    return


@app.cell
def _(prob_value):
    prob_value
    return


@app.cell
def _(mo):
    # probability checker
    prob_value = mo.ui.number(
        value=0.5,
        start=-1,
        stop=100,
        label="Enter a probability"
    )
    return (prob_value,)


@app.cell
def _(mo, prob_value):
    is_valid = 0 <= prob_value.value <= 1

    message = mo.md(f"""
    ### Is this a valid probability?

    **Value checked**: {prob_value.value}

    **Result**: {"✅ Valid" if is_valid else "❌ Invalid"}

    **Why?** {"This is between 0 and 1" if is_valid else "Probabilities must be between 0 and 1"}
    """)

    mo.callout(message, kind="success" if is_valid else "danger")
    return is_valid, message


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🌍 Second Axiom: Total Probability

        The second axiom states that the probability of all possible outcomes together must equal 1.

        $P(S) = 1$ where S is the sample space (all possible outcomes)

        Let's verify this with a weather forecast!
        """
    )
    return


@app.cell
def _(mo, rainy_prob, sunny_prob):
    mo.hstack([sunny_prob, rainy_prob]).center()
    return


@app.cell
def _(mo):
    # probability adjuster
    sunny_prob = mo.ui.slider(
        value=0.6,
        start=0,
        stop=1,
        step=0.1,
        label="Probability of Sunny ☀️"
    )
    rainy_prob = mo.ui.slider(
        value=0.3,
        start=0,
        stop=1,
        step=0.1,
        label="Probability of Rainy 🌧️"
    )
    return rainy_prob, sunny_prob


@app.cell
def _(mo, rainy_prob, sunny_prob):
    total_prob = sunny_prob.value + rainy_prob.value
    cloudy_prob = 1 - total_prob if total_prob <= 1 else 0

    weather_status = mo.md(f"""
    ### Weather Probability Check

    ☀️ Sunny: {sunny_prob.value}
    🌧️ Rainy: {rainy_prob.value}
    ☁️ Cloudy: {cloudy_prob:.1f}

    **Total**: {total_prob + cloudy_prob:.1f}

    **Status**: {"✅ Valid" if abs(total_prob + cloudy_prob - 1) < 0.01 else "❌ Invalid"}
    """)

    mo.callout(
        weather_status,
        kind="success" if abs(total_prob + cloudy_prob - 1) < 0.01 else "danger"
    )
    return cloudy_prob, total_prob, weather_status


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎮 Let's Practice!

        Try this simple exercise to check your understanding:
        """
    )
    return


@app.cell
def _(question):
    question
    return


@app.cell
def _(answer, check_button, mo):
    mo.hstack([answer, check_button], justify="start")
    return


@app.cell
def _(mo, random):
    # Generate a simple probability question
    correct_prob = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])

    question = mo.md(f"""
    If P(Heads) = {correct_prob}, what must P(Tails) be?
    """)

    answer = mo.ui.number(
        value=0.5,
        start=0,
        stop=1,
        step=0.1,
        label="Your answer"
    )

    check_button = mo.ui.button(label="Check Answer")
    return answer, check_button, correct_prob, question


@app.cell
def _(answer, check_button, correct_prob, mo):
    check_callout = None
    if check_button.value:
        correct_answer = 1 - correct_prob
        is_correct = abs(answer.value - correct_answer) < 0.01

        result = mo.md(f"""
        ### Your Answer: {answer.value}

        **Correct Answer**: {correct_answer}

        {"🎉 Perfect! The probabilities sum to 1" if is_correct 
         else "❌ Remember: Probabilities must sum to 1"}
        """)
        mo.callout(result, kind="success" if is_correct else "danger")
    check_callout
    return check_callout, correct_answer, is_correct, result


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

    - Sample Space and Events
    - Probability Functions
    - Addition Laws of Probability
        - Simple Events
        - Mutually Exclusive Events

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


@app.cell
def _():
    import random
    return (random,)


if __name__ == "__main__":
    app.run()

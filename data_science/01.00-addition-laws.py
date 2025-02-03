# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "matplotlib-venn"
# ]
# ///

import marimo

__generated_with = "0.10.19"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🎲 Addition Laws of Probability

        Welcome to the world of combining probabilities! In this notebook, we'll learn how to:

        - Add probabilities of simple events

        - Work with mutually exclusive events

        - Use the general addition rule

        Let's make probability addition fun and visual! 🎯
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎯 Simple Events

        Simple events are basic outcomes that don't overlap. Think of rolling a die:

        - Rolling a 1 is a simple event

        - Rolling a 2 is another simple event

        - They can't happen at the same time!
        """
    )
    return


@app.cell
def _(die_events):
    die_events
    return


@app.cell
def _(mo):
    # Interactive die probability calculator
    die_events = mo.ui.multiselect(
        options=["1", "2", "3", "4", "5", "6"],
        value=["1", "2"],
        label="Select die numbers"
    )
    return (die_events,)


@app.cell
def _(die_events, mo, plt):
    selected = [int(x) for x in die_events.value]
    die_prob = len(selected) / 6

    # Create simple bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(['Selected', 'Others'], 
            [die_prob, 1-die_prob],
            color=['#2ecc71', '#e74c3c'])
    plt.title('Probability Distribution')
    plt.ylabel('Probability')
    plt.ylim(0, 1)

    die_result = mo.md(f"""
    ### Probability Analysis

    Selected numbers: {', '.join(die_events.value)}

    P(selected) = {die_prob:.3f}

    This is a simple addition because each number can only come up once!
    """)

    mo.hstack([plt.gcf(), die_result])
    return die_prob, die_result, selected


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🚫 Mutually Exclusive Events

        Events are mutually exclusive when they cannot happen at the same time.

        For mutually exclusive events A and B:

        $P(A \cup B) = P(A) + P(B)$

        Let's visualize this with a weather example!
        """
    )
    return


@app.cell
def _(mo, rainy_prob, sunny_prob):
    mo.hstack([sunny_prob, rainy_prob])
    return


@app.cell
def _(mo):
    # Weather probability selector
    sunny_prob = mo.ui.slider(
        value=0.3,
        start=0,
        stop=1,
        step=0.1,
        label="P(Sunny)"
    )
    rainy_prob = mo.ui.slider(
        value=0.4,
        start=0,
        stop=1,
        step=0.1,
        label="P(Rainy)"
    )
    return rainy_prob, sunny_prob


@app.cell
def _(mo, plt, rainy_prob, sunny_prob, venn2):
    weather_total = sunny_prob.value + rainy_prob.value

    # Create Venn diagram
    plt.figure(figsize=(8, 5))
    v = venn2(subsets=(sunny_prob.value, rainy_prob.value, 0),
              set_labels=('Sunny', 'Rainy'))

    # Customize colors
    if v:
        v.get_patch_by_id('10').set_color('#f1c40f')
        v.get_patch_by_id('01').set_color('#3498db')

    plt.title('Mutually Exclusive Weather Events')

    weather_result = mo.md(f"""
    ### Weather Probability Analysis

    P(Sunny) = {sunny_prob.value}
    P(Rainy) = {rainy_prob.value}
    P(Sunny or Rainy) = {weather_total}

    Status: {"✅ Valid" if weather_total <= 1 else "❌ Invalid - Total exceeds 1!"}
    """)

    mo.hstack([plt.gcf(), weather_result])
    return v, weather_result, weather_total


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🤝 General Addition Rule

        When events can overlap, we need to subtract the overlap:

        $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

        Let's explore this with a student club example!
        """
    )
    return


@app.cell
def _(math_prob, mo, overlap_prob, science_prob):
    mo.hstack([math_prob, science_prob, overlap_prob])
    return


@app.cell
def _(mo):
    # Student club membership
    math_prob = mo.ui.slider(
        value=0.4,
        start=0,
        stop=1,
        step=0.1,
        label="P(Math Club)"
    )
    science_prob = mo.ui.slider(
        value=0.3,
        start=0,
        stop=1,
        step=0.1,
        label="P(Science Club)"
    )
    overlap_prob = mo.ui.slider(
        value=0.1,
        start=0,
        stop=0.3,
        step=0.1,
        label="P(Both Clubs)"
    )
    return math_prob, overlap_prob, science_prob


@app.cell
def _(math_prob, mo, overlap_prob, plt, science_prob, venn2):
    # Calculate total probability
    club_total = math_prob.value + science_prob.value - overlap_prob.value

    # Create Venn diagram
    plt.figure(figsize=(8, 5))
    _v = venn2(subsets=(
        math_prob.value - overlap_prob.value,
        science_prob.value - overlap_prob.value,
        overlap_prob.value
    ), set_labels=('Math', 'Science'))

    plt.title('Club Membership Overlap')

    club_result = mo.md(f"""
    ### Club Membership Analysis

    P(Math) = {math_prob.value}
    P(Science) = {science_prob.value}
    P(Both) = {overlap_prob.value}

    P(Math or Science) = {club_total:.2f}

    Using the formula:
    {math_prob.value} + {science_prob.value} - {overlap_prob.value} = {club_total:.2f}
    """)

    mo.hstack([plt.gcf(), club_result])
    return club_result, club_total


@app.cell
def _(random):
    # Generate random percentages that make sense
    activity1_percent = random.randint(30, 70)
    activity2_percent = random.randint(30, 70)
    return activity1_percent, activity2_percent


@app.cell
def _(practice_prompt):
    practice_prompt
    return


@app.cell(hide_code=True)
def _(activity1_percent, activity2_percent, mo):
    practice_prompt = mo.md(f"""
    ## 🎯 Practice Time!

    A survey of students found:

    - {activity1_percent}% play video games

    - {activity2_percent}% play sports

    What's the maximum possible percentage that could do either activity?
    What's the minimum?

    Think about:

    - What happens if there's no overlap?

    - What happens if one group completely contains the other?
    """)
    return (practice_prompt,)


@app.cell
def _(max_answer, min_answer, mo):
    mo.hstack([min_answer, max_answer])
    return


@app.cell
def _(check):
    check
    return


@app.cell
def _(mo):
    max_answer = mo.ui.number(value=0, start=0, stop=100, label="Maximum %")
    min_answer = mo.ui.number(value=0, start=0, stop=100, label="Minimum %")
    return max_answer, min_answer


@app.cell
def _(mo):
    check = mo.ui.run_button()
    return (check,)


@app.cell
def _(
    activity1_percent,
    activity2_percent,
    check,
    max_answer,
    min_answer,
    mo,
):
    callout1, callout2 = None, None
    if check.value:
        # Calculate correct answers
        correct_max = min(100, activity1_percent + activity2_percent)
        correct_min = max(activity1_percent, activity2_percent)

        max_correct = max_answer.value == correct_max
        min_correct = min_answer.value == correct_min

        practice_explanation = mo.md(f"""
        ### Answer Analysis

        **Maximum**: {max_answer.value}% - {"✅ Correct!" if max_correct else f"❌ Should be {correct_max}%"}
        
        - Maximum occurs when there's no overlap
        
        - {activity1_percent}% + {activity2_percent}% = {activity1_percent + activity2_percent}%
        {f"- But can't exceed 100%, so maximum is 100%" if activity1_percent + activity2_percent > 100 else ""}

        **Minimum**: {min_answer.value}% - {"✅ Correct!" if min_correct else f"❌ Should be {correct_min}%"}
        
        - Minimum occurs when one group completely contains the other
        
        - max({activity1_percent}%, {activity2_percent}%) = {correct_min}%
        """)

        callout1 = mo.callout(practice_explanation, kind="success" if (max_correct and min_correct) else "danger")

        # Add a visual representation
        if max_correct and min_correct:
            bonus_insight = mo.md(f"""
            ### 🌟 Bonus Insight

            The actual percentage must be somewhere between:
            {correct_min}% ≤ P(Video Games ∪ Sports) ≤ {correct_max}%

            This depends on how much overlap exists between the groups!
            """)
            callout2 = mo.callout(bonus_insight, kind="info")
    callout1
    return (
        bonus_insight,
        callout1,
        callout2,
        correct_max,
        correct_min,
        max_correct,
        min_correct,
        practice_explanation,
    )


@app.cell(hide_code=True)
def _(mo):
    key_points = mo.md("""
    ## 🎯 Key Takeaways

    1. Simple events: Just add the probabilities
    2. Mutually exclusive: P(A ∪ B) = P(A) + P(B)
    3. General case: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    """)

    next_topics = mo.md("""
    ## 📚 Coming Up Next

    - Multiplication Laws
        - Independent Events
        - Conditional Probability
        - Chain Rule

    Get ready for more probability adventures! 🚀
    """)

    mo.hstack([
        mo.callout(key_points, kind="info"),
        mo.callout(next_topics, kind="success")
    ])
    return key_points, next_topics


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2, venn2_circles
    import numpy as np
    import random
    return mo, np, plt, random, venn2, venn2_circles


if __name__ == "__main__":
    app.run()

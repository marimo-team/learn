# /// script
# requires-python = "==3.12"
# dependencies = [
#     "anywidget",
#     "marimo",
#     "marimo-learn",
# ]
# ///

"""
Example Marimo Notebook: Education Widgets Demo

This notebook demonstrates all formative assessment widgets.
Run with: marimo edit demo.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from marimo_learn import (
        Color,
        ConceptMapWidget,
        FlashcardWidget,
        LabelingWidget,
        MatchingWidget,
        MultipleChoiceWidget,
        NumericEntryWidget,
        OrderingWidget,
        PredictThenCheckWidget,
        World,
    )

    return (
        Color,
        ConceptMapWidget,
        FlashcardWidget,
        LabelingWidget,
        MatchingWidget,
        MultipleChoiceWidget,
        NumericEntryWidget,
        OrderingWidget,
        PredictThenCheckWidget,
        World,
    )


@app.cell
def _(mo):
    mo.md("""
    # Educational Widgets Demo

    This notebook demonstrates widgets in marimo_learn.
    """)
    return


# Spiral
@app.cell
def _(Color, World, mo):
    _world = World()

    async def _spiral(world, turtle):
        colors = list(Color)
        for i in range(70):
            if i % 10 == 0:
                turtle.set_color(colors[(i // 10) % len(colors)])
            await turtle.forward(i * 2.8)
            turtle.right(91)

    _world.set_coroutine(_spiral)
    mo.ui.anywidget(_world)


@app.cell
def _(mo):
    mo.md("""
    ## Concept Map
    """)
    return


@app.cell
def _(ConceptMapWidget, mo):
    concept_map = mo.ui.anywidget(
        ConceptMapWidget(
            question="Map the relationships between these Python concepts:",
            concepts=["function", "parameter", "argument", "return value", "call site"],
            terms=["defines", "accepts", "supplies", "produces", "invokes"],
            correct_edges=[
                {"from": "function", "to": "parameter", "label": "accepts"},
                {"from": "function", "to": "return value", "label": "produces"},
                {"from": "call site", "to": "argument", "label": "supplies"},
                {"from": "argument", "to": "parameter", "label": "defines"},
                {"from": "call site", "to": "function", "label": "invokes"},
            ],
        )
    )
    concept_map
    return (concept_map,)


@app.cell
def _(concept_map, mo):
    def concept_map_msg(widget):
        val = widget.value.get("value") or {}
        score = val.get("score")
        total = val.get("total", 5)
        if score is None:
            return "Draw connections between concepts to see your score."
        msg = f"**{score}/{total}** correct connection{'s' if total != 1 else ''}"
        if val.get("correct"):
            msg += " — complete!"
        return msg

    mo.md(concept_map_msg(concept_map))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Flashcard Deck
    """)
    return


@app.cell
def _(FlashcardWidget, mo):
    flashcard_deck = mo.ui.anywidget(
        FlashcardWidget(
            question="Python Concepts — rate yourself on each card:",
            cards=[
                {
                    "front": "What does a list comprehension look like?",
                    "back": "[expr for item in iterable if condition] — e.g., [x**2 for x in range(10) if x % 2 == 0]",
                },
                {
                    "front": "What is the difference between a list and a tuple?",
                    "back": "Lists are mutable (can be changed after creation); tuples are immutable (cannot be changed).",
                },
                {
                    "front": "What does the `*args` parameter do in a function definition?",
                    "back": "It collects any number of positional arguments into a tuple named `args`.",
                },
                {
                    "front": "What is a Python generator?",
                    "back": "A function that uses `yield` to produce values one at a time, pausing between each, without building the full sequence in memory.",
                },
                {
                    "front": "What does `if __name__ == '__main__':` do?",
                    "back": "It runs the indented code only when the file is executed directly, not when it is imported as a module.",
                },
                {
                    "front": "What is the difference between `is` and `==` in Python?",
                    "back": "`==` tests value equality; `is` tests identity (whether two names refer to the exact same object in memory).",
                },
            ],
            shuffle=True,
        )
    )
    flashcard_deck
    return (flashcard_deck,)


@app.cell
def _(flashcard_deck, mo):
    def flashcard_progress(widget):
        val = widget.value.get("value") or {}
        results = val.get("results", {})
        counts = {"got_it": 0, "almost": 0, "no": 0}
        for r in results.values():
            counts[r["rating"]] = counts.get(r["rating"], 0) + 1
        return len(results), counts, val.get("complete", False)

    _rated, _counts, _complete = flashcard_progress(flashcard_deck)
    mo.md(f"""
    **Progress:** {_rated} card(s) rated —
    ✓ Got it: {_counts["got_it"]} &nbsp;
    ~ Almost: {_counts["almost"]} &nbsp;
    ✗ No: {_counts["no"]}
    {"&nbsp; 🎉 Deck complete!" if _complete else ""}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Labeling Question
    """)
    return


@app.cell
def _(LabelingWidget, mo):
    labeling_question = mo.ui.anywidget(
        LabelingWidget(
            question="Label the parts of this Python code:",
            labels=[
                "Variable declaration",
                "Function call",
                "String literal",
                "Arithmetic operation",
            ],
            text_lines=[
                "name = 'Alice'",
                "age = 25",
                "result = age + 5",
                "print(name)",
            ],
            correct_labels={
                0: [0, 2],  # Line 0: labels 0 and 2
                1: [0],  # Line 1: label 0
                2: [0, 3],  # Line 2: labels 0 and 3
                3: [1],  # Line 3: label 1
            },
        )
    )
    labeling_question
    return (labeling_question,)


@app.cell
def _(labeling_question, mo):
    _val = labeling_question.value.get("value") or {}
    _total = _val.get("total", 0)
    _msg = f"Score: {_val['score']}/{_total}" if _total > 0 else "Not submitted yet"
    mo.md(f"**{_msg}**")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Matching Question
    """)
    return


@app.cell
def _(MatchingWidget, mo):
    matching_question = mo.ui.anywidget(
        MatchingWidget(
            question="Match the programming languages to their primary paradigms:",
            left=["Python", "Haskell", "C", "SQL"],
            right=["Functional", "Procedural", "Multi-paradigm", "Declarative"],
            correct_matches={0: 2, 1: 0, 2: 1, 3: 3},
        )
    )
    matching_question
    return (matching_question,)


@app.cell
def _(matching_question, mo):
    _val = matching_question.value.get("value") or {}
    _msg = f"Score: {_val['score']}/{_val['total']}" if _val else "Not answered yet"
    mo.md(f"**{_msg}**")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Multiple Choice Question
    """)
    return


@app.cell
def _(MultipleChoiceWidget, mo):
    multiple_choice_question = mo.ui.anywidget(
        MultipleChoiceWidget(
            question="What is the capital of France?",
            options=["London", "Berlin", "Paris", "Madrid"],
            correct_answer=2,
            explanation="Paris has been the capital of France since the 12th century.",
        )
    )
    multiple_choice_question
    return (multiple_choice_question,)


@app.cell
def _(multiple_choice_question, mo):
    _val = multiple_choice_question.value.get("value") or {}
    if _val.get("answered"):
        _msg = "Score: 1/1" if _val.get("correct") else "Score: 0/1"
    else:
        _msg = "Not answered yet"
    mo.md(f"**{_msg}**")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Ordering Question
    """)
    return


@app.cell
def _(OrderingWidget, mo):
    ordering_question = mo.ui.anywidget(
        OrderingWidget(
            question="Arrange these steps of the scientific method in the correct order:",
            items=[
                "Ask a question",
                "Do background research",
                "Construct a hypothesis",
                "Test with an experiment",
                "Analyze data",
                "Draw conclusions",
            ],
            shuffle=True,
        )
    )
    ordering_question
    return (ordering_question,)


@app.cell
def _(mo, ordering_question):
    _val = ordering_question.value.get("value") or {}
    if _val.get("correct") is not None and _val.get("order"):
        _msg = "Score: 1/1" if _val.get("correct") else "Score: 0/1"
    else:
        _msg = "Not answered yet"
    mo.md(f"**{_msg}**")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Numeric Entry Question
    """)
    return


@app.cell
def _(NumericEntryWidget, mo):
    numeric_entry_question = mo.ui.anywidget(
        NumericEntryWidget(
            question="How many bits are in one byte?",
            correct_answer=8,
            tolerance=0.5,
            explanation="A byte consists of exactly 8 bits.",
        )
    )
    numeric_entry_question
    return (numeric_entry_question,)


@app.cell
def _(mo, numeric_entry_question):
    _val = numeric_entry_question.value.get("value") or {}
    if _val.get("answered"):
        _msg = "Score: 1/1" if _val.get("ok") else "Score: 0/1"
    else:
        _msg = "Not answered yet"
    mo.md(f"**{_msg}**")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Predict-Then-Check Question
    """)
    return


@app.cell
def _(PredictThenCheckWidget, mo):
    predict_then_check_question = mo.ui.anywidget(
        PredictThenCheckWidget(
            question="What does this Python code print?",
            code='words = ["one", "two", "three"]\nprint(len(words))',
            output="3",
            options=["2", "3", "['one', 'two', 'three']", "None"],
            correct_answer=1,
            explanations=[
                "Wrong: len() counts all items; the list has three elements.",
                "Correct: len() returns the number of items, which is 3.",
                "Wrong: len() returns an integer count, not the list itself.",
                "Wrong: len() always returns an integer, never None.",
            ],
        )
    )
    predict_then_check_question
    return (predict_then_check_question,)


@app.cell
def _(mo, predict_then_check_question):
    _val = predict_then_check_question.value.get("value") or {}
    if _val.get("answered"):
        _msg = "Score: 1/1" if _val.get("correct") else "Score: 0/1"
    else:
        _msg = "Not answered yet"
    mo.md(f"**{_msg}**")
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Quiz Results Summary

    You can access the values from all widgets to create a summary or scoring system.
    """)
    return


@app.cell
def _(
    concept_map,
    flashcard_deck,
    matching_question,
    multiple_choice_question,
    mo,
    numeric_entry_question,
    ordering_question,
    predict_then_check_question,
):
    def widget_val(widget):
        return widget.value.get("value") or {}

    def calculate_score():
        score = 0
        total = 0
        for val, answered_key, correct_key in [
            (widget_val(concept_map), "score", "correct"),
            (widget_val(matching_question), "score", "correct"),
            (widget_val(multiple_choice_question), "answered", "correct"),
            (widget_val(ordering_question), "order", "correct"),
            (widget_val(numeric_entry_question), "answered", "ok"),
            (widget_val(predict_then_check_question), "answered", "correct"),
        ]:
            if val.get(answered_key) is not None and val.get(answered_key) is not False:
                total += 1
                if val.get(correct_key):
                    score += 1
        fc = widget_val(flashcard_deck)
        if fc.get("results"):
            total += 1
            if fc.get("complete"):
                score += 1
        return score, total

    score, total = calculate_score()
    mo.md(f"""
    ### Current Score: {score}/{total}

    {"🎉 Perfect score!" if score == total and total > 0 else "Keep going!" if total > 0 else "Answer the questions above to see your score."}
    """)
    return


if __name__ == "__main__":
    app.run()

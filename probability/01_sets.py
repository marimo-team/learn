# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sets

    Probability is the study of "events", assigning numerical values to how likely
    events are to occur. For example, probability lets us quantify how likely it is for it to rain or shine on a given day.


    Typically we reason about _sets_ of events. In mathematics,
    a set is a collection of elements, with no element included more than once.
    Elements can be any kind of object.

    For example:

    - ☀️ Weather events: $\{\text{Rain}, \text{Overcast}, \text{Clear}\}$
    - 🎲 Die rolls: $\{1, 2, 3, 4, 5, 6\}$
    - 🪙 Pairs of coin flips = $\{ \text{(Heads, Heads)}, \text{(Heads, Tails)}, \text{(Tails, Tails)} \text{(Tails, Heads)}\}$

    Sets are the building blocks of probability, and will arise frequently in our study.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set operations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In Python, sets are made with the `set` function:
    """)
    return


@app.cell
def _():
    A = set([2, 3, 5, 7])
    A
    return (A,)


@app.cell
def _():
    B = set([0, 1, 2, 3, 5, 8])
    B
    return (B,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below we explain common operations on sets.

    _**Try it!** Try modifying the definitions of `A` and `B` above, and see how the results change below._

    The **union** $A \cup B$ of sets $A$ and $B$ is the set of elements in $A$, $B$, or both.
    """)
    return


@app.cell
def _(A, B):
    A | B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **intersection** $A \cap B$ is the set of elements in both $A$ and $B$
    """)
    return


@app.cell
def _(A, B):
    A & B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **difference** $A \setminus B$ is the set of elements in $A$ that are not in $B$.
    """)
    return


@app.cell
def _(A, B):
    A - B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 🎬 An interactive example

    Here's a simple example that classifies TV shows into sets by genre, and uses these sets to recommend shows to a user based on their preferences.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    viewer_type = mo.ui.radio(
        options={
            "I like action and drama!": "New Viewer",
            "I only like action shows": "Action Fan",
            "I only like dramas": "Drama Fan",
        },
        value="I like action and drama!",
        label="Which genre do you prefer?",
    )
    return (viewer_type,)


@app.cell(hide_code=True)
def _(viewer_type):
    viewer_type
    return


@app.cell
def _():
    action_shows = {"Stranger Things", "The Witcher", "Money Heist"}
    drama_shows = {"The Crown", "Money Heist", "Bridgerton"}
    return action_shows, drama_shows


@app.cell
def _(action_shows, drama_shows):
    recommendations = {
        "New Viewer": action_shows | drama_shows,  # Union for new viewers
        "Action Fan": action_shows - drama_shows,  # Unique action shows
        "Drama Fan": drama_shows - action_shows,  # Unique drama shows
    }
    return (recommendations,)


@app.cell(hide_code=True)
def _(mo, recommendations, viewer_type):
    result = recommendations[viewer_type.value]

    explanation = {
        "New Viewer": "You get everything to explore!",
        "Action Fan": "Pure action, no drama!",
        "Drama Fan": "Drama-focused selections!",
    }

    mo.md(f"""
    **🎬 Recommended shows.** Based on your preference for **{viewer_type.value}**,
    we recommend:

    {", ".join(result)}

    **Why these shows?** 
    {explanation[viewer_type.value]}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Exercise

    Given these sets:

    - A = {🎮, 📱, 💻}

    - B = {📱, 💻, 🖨️}

    - C = {💻, 🖨️, ⌨️}

    Can you:

    1. Find all elements that are in A or B

    2. Find elements common to all three sets

    3. Find elements in A that aren't in C

    <details>

    <summary>Check your answers!</summary>

    1. A ∪ B = {🎮, 📱, 💻, 🖨️}<br>
    2. A ∩ B ∩ C = {💻}<br>
    3. A - C = {🎮, 📱}

    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧮 Set properties

    Here are some important properties of the set operations:

    1. **Commutative**: $A \cup B = B \cup A$
    2. **Associative**: $(A \cup B) \cup C = A \cup (B \cup C)$
    3. **Distributive**: $A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set builder notation

    To compactly describe the elements in a set, we can use **set builder notation**, which specifies conditions that must be true for elements to be in the set.

    For example, here is how to specify the set of positive numbers less than 10:

    \[
    \{x \mid 0 < x < 10 \}
    \]

    The predicate to the right of the vertical bar $\mid$ specifies conditions that must be true for an element to be in the set; the expression to the left of $\mid$ specifies the value being included.

    In Python, set builder notation is called a "set comprehension."
    """)
    return


@app.function
def predicate(x):
    return x > 0 and x < 10


@app.cell
def _():
    set(x for x in range(100) if predicate(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Try it!** Try modifying the `predicate` function above and see how the set changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    You've learned:

    - Basic set operations
    - Set properties
    - Real-world applications

    In the next lesson, we'll define probability from the ground up, using sets.

    Remember: In probability, every event is a set, and every set can be an event!
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

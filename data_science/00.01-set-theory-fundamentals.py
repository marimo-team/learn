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
    mo.md(
        """
        # 🎯 Set Theory: The Building Blocks of Probability

        Welcome to the magical world of sets! Think of sets as the LEGO® blocks of probability - 
        they're the fundamental pieces we use to build more complex concepts.

        ## What is a Set? 

        A set is a collection of distinct objects, called elements or members. 

        For example:

        - 🎨 Colors = {red, blue, green}

        - 🔢 Even numbers under 10 = {2, 4, 6, 8}

        - 🐾 Pets = {dog, cat, hamster, fish}
        """
    )
    return


@app.cell
def _(elements):
    elements
    return


@app.cell(hide_code=True)
def _(mo):
    # interactive set creator
    elements = mo.ui.text(
        value="🐶,🐱,🐹",
        label="Create your own set (use commas to separate elements)"
    )
    return (elements,)


@app.cell(hide_code=True)
def _(elements, mo):
    user_set = set(elements.value.split(','))

    mo.md(f"""
    ### Your Custom Set:

    ${{{', '.join(user_set)}}}$

    Number of elements: {len(user_set)}
    """)
    return (user_set,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎮 Set Operations Playground

        Let's explore the three fundamental set operations:

        - Union (∪): Combining sets

        - Intersection (∩): Finding common elements

        - Difference (-): What's unique to one set

        Try creating two sets below!
        """
    )
    return


@app.cell
def _(operation):
    operation
    return


@app.cell(hide_code=True)
def _(mo):
    set_a = mo.ui.text(value="🍕,🍔,🌭,🍟", label="Set A (Fast Food)")
    set_b = mo.ui.text(value="🍕,🥗,🥙,🍟", label="Set B (Healthy-ish Food)")

    operation = mo.ui.dropdown(
        options=["Union", "Intersection", "Difference"],
        value="Union",
        label="Choose Operation"
    )
    return operation, set_a, set_b


@app.cell
def _(mo, operation, set_a, set_b):
    A = set(set_a.value.split(','))
    B = set(set_b.value.split(','))

    results = {
        "Union": (A | B, "∪", "Everything from both sets"),
        "Intersection": (A & B, "∩", "Common elements"),
        "Difference": (A - B, "-", "In A but not in B")
    }

    _result, symbol, description = results[operation.value]

    mo.md(f"""
    ### Set Operation Result

    $A {symbol} B = {{{', '.join(_result)}}}$

    **What this means**: {description}

    **Set A**: {', '.join(A)}
    **Set B**: {', '.join(B)}
    """)
    return A, B, description, results, symbol


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎬 Netflix Shows Example

        Let's use set theory to understand content recommendations!
        """
    )
    return


@app.cell
def _(viewer_type):
    viewer_type
    return


@app.cell
def _(mo):
    # Netflix genres example
    action_fans = {"Stranger Things", "The Witcher", "Money Heist"}
    drama_fans = {"The Crown", "Money Heist", "Bridgerton"}

    viewer_type = mo.ui.radio(
        options=["New Viewer", "Action Fan", "Drama Fan"],
        value="New Viewer",
        label="Select Viewer Type"
    )
    return action_fans, drama_fans, viewer_type


@app.cell
def _(action_fans, drama_fans, mo, viewer_type):
    recommendations = {
        "New Viewer": action_fans | drama_fans,  # Union for new viewers
        "Action Fan": action_fans - drama_fans,  # Unique action shows
        "Drama Fan": drama_fans - action_fans    # Unique drama shows
    }

    result = recommendations[viewer_type.value]

    explanation = {
        "New Viewer": "You get everything to explore!",
        "Action Fan": "Pure action, no drama!",
        "Drama Fan": "Drama-focused selections!"
    }

    mo.md(f"""
    ### 🎬 Recommended Shows

    Based on your preference for **{viewer_type.value}**, we recommend:

    {', '.join(result)}

    **Why these shows?** 
    {explanation[viewer_type.value]}
    """)
    return explanation, recommendations, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🧮 Set Properties

        Important properties of sets:

        1. **Commutative**: A ∪ B = B ∪ A
        2. **Associative**: (A ∪ B) ∪ C = A ∪ (B ∪ C)
        3. **Distributive**: A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)

        Let's verify these with a fun exercise!
        """
    )
    return


@app.cell
def _(mo, property_check, set_size):
    mo.hstack([property_check, set_size])
    return


@app.cell
def _(mo):
    # Interactive property verifier
    property_check = mo.ui.dropdown(
        options=[
            "Commutative (Union)",
            "Commutative (Intersection)",
            "Associative (Union)"
        ],
        value="Commutative (Union)",
        label="Select Property to Verify"
    )

    set_size = mo.ui.slider(
        value=3,
        start=2,
        stop=5,
        label="Set Size for Testing"
    )
    return property_check, set_size


@app.cell
def _(mo, property_check, set_size):
    import random

    # Create random emoji sets for verification
    emojis = ["😀", "😎", "🤓", "🤠", "😴", "🤯", "🤪", "😇"]

    set1 = set(random.sample(emojis, set_size.value))
    set2 = set(random.sample(emojis, set_size.value))

    operations = {
        "Commutative (Union)": (
            set1 | set2,
            set2 | set1,
            "A ∪ B = B ∪ A"
        ),
        "Commutative (Intersection)": (
            set1 & set2,
            set2 & set1,
            "A ∩ B = B ∩ A"
        ),
        "Associative (Union)": (
            (set1 | set2) | set(random.sample(emojis, set_size.value)),
            set1 | (set2 | set(random.sample(emojis, set_size.value))),
            "(A ∪ B) ∪ C = A ∪ (B ∪ C)"
        )
    }

    result1, result2, formula = operations[property_check.value]

    mo.md(f"""
    ### Property Verification

    **Testing**: {formula}

    Set A: {', '.join(set1)}
    Set B: {', '.join(set2)}

    **Left Side**: {', '.join(result1)}
    **Right Side**: {', '.join(result2)}

    **Property holds**: {'✅' if result1 == result2 else '❌'}
    """)
    return emojis, formula, operations, random, result1, result2, set1, set2


@app.cell(hide_code=True)
def _(mo):
    quiz = mo.md("""
    ## 🎯 Quick Challenge

    Given these sets:

    - A = {🎮, 📱, 💻}

    - B = {📱, 💻, 🖨️}

    - C = {💻, 🖨️, ⌨️}

    Can you:

    1. Find all elements that are in either A or B

    2. Find elements common to all three sets

    3. Find elements in A that aren't in C

    <details>

    <summary>Check your answers!</summary>

    1. A ∪ B = {🎮, 📱, 💻, 🖨️}<br>
    2. A ∩ B ∩ C = {💻}<br>
    3. A - C = {🎮, 📱}

    </details>
    """)

    mo.callout(quiz, kind="info")
    return (quiz,)


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## 🎯 Set Theory Master in Training!

    You've learned:

    - Basic set operations

    - Set properties

    - Real-world applications

    Coming up next: Axiomatic Probability! 🎲✨

    Remember: In probability, every event is a set, and every set can be an event!
    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

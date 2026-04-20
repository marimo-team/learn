# /// script
# requires-python = "==3.12"
# dependencies = [
#     "anywidget",
#     "marimo",
#     "marimo-learn==0.13.0",
# ]
# ///

"""
Example Marimo Notebook: Turtle Graphics
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from marimo_learn import (
        Color,
        World,
    )

    return Color, World, mo


@app.cell
def _(mo):
    mo.md("""
    # Turtle Graphics

    The [marimo-learn](https://pypi.org/project/marimo-learn/) package includes a simple implementation of turtle graphics. Learners can create worlds, and then write functions to make a turtle draw multi-colored shapes. Crucially, this widget was created in only 15 minutes with the assistance of an LLM, which shows just how easy it now is for educators to create tools that meet their specific needs.
    """)
    return


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
    return


if __name__ == "__main__":
    app.run()

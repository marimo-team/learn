# /// script
# requires-python = "==3.12"
# dependencies = [
#     "anywidget",
#     "marimo",
#     "marimo-learn",
# ]
# ///

"""
Example Marimo Notebook: Turtle Graphics
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from marimo_learn import (
        Color,
        World,
    )

    return (
        Color,
        World,
    )


@app.cell
def _(mo):
    mo.md("""
    # Turtle Graphics Demo

    This notebook demonstrates turtle graphics in marimo_learn.
    See [the documentation](https://github.com/gvwilson/marimo_learn) for details.
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


if __name__ == "__main__":
    app.run()

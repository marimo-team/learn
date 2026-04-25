"""Turtle and World classes."""

import asyncio
from collections.abc import Callable
from enum import Enum
import math
from pathlib import Path
import time

import anywidget
import traitlets

# Default canvas dimensions
WIDTH = 480
HEIGHT = 480

# Render rate: seconds between frames
DEFAULT_DELAY = 0.05

# Initial heading: -90° points upward in SVG coordinate space
INITIAL_ANGLE = -90.0

# SVG line thickness for drawn segments
STROKE_WIDTH = 1.8

# Size of the equilateral triangle that represents the turtle cursor
TURTLE_RADIUS = 9

# Turtle cursor appearance
TURTLE_COLOR = "#00ff88"
TURTLE_OPACITY = 0.9

# Canvas background and border
BACKGROUND_COLOR = "#1a1a2e"
BORDER_RADIUS = 8


class Color(str, Enum):
    """Standard colors available to Turtle turtles."""

    CORNFLOWER = "#8ecae6"
    CRIMSON = "#e63946"
    GOLD = "#e9c46a"
    SAGE = "#b5e48c"
    SANDY = "#f4a261"
    SKY = "#a8dadc"
    TEAL = "#2ec4b6"


class World(anywidget.AnyWidget):
    """
    Canvas widget that owns rendering and hosts one or more turtles.

    Typical notebook usage::

        world = World()

        async def my_drawing(world, turtle):
            for i in range(60):
                await turtle.forward(i * 3)
                turtle.right(91)

        world.set_coroutine(my_drawing)
        mo.ui.anywidget(world)   # display via mo.ui.anywidget for live comm

    Drawing runs as an asyncio task in Marimo's event loop, so the kernel
    stays free and Stop works immediately.

    For testing, pass ``output_fn`` to bypass widget rendering::

        world = World(output_fn=lambda _: None)
        await world.run(my_drawing())
    """

    _esm = Path(__file__).parent / "static" / "turtle.js"

    width = traitlets.Int(WIDTH).tag(sync=True)
    height = traitlets.Int(HEIGHT).tag(sync=True)
    delay = traitlets.Float(DEFAULT_DELAY).tag(sync=True)
    # Render state pushed to JS on each frame: {segments, turtles, done, ts}
    _render = traitlets.Dict({}).tag(sync=True)
    # Incremented by JS each time Start is pressed
    _start_counter = traitlets.Int(0).tag(sync=True)
    # Set True by JS when Stop is pressed; Python clears after handling
    _stop_requested = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        width: int = WIDTH,
        height: int = HEIGHT,
        delay: float = DEFAULT_DELAY,
        output_fn: Callable[[str], None] | None = None,
    ):
        super().__init__(width=width, height=height, delay=delay)
        self._turtles: list["Turtle"] = []
        self._dirty = False
        self._last_render: float = 0.0
        self._stop = False
        self._coro_fns: list = []
        # output_fn is used in test / non-widget mode; None means widget mode
        self._output_fn = output_fn

    def turtle(self) -> "Turtle":
        """Create a new turtle that belongs to this world."""
        t = Turtle(self)
        self._turtles.append(t)
        return t

    def set_coroutine(self, *coro_fns) -> None:
        """Register async drawing functions to run when Start is pressed.

        Each function must accept ``(world, turtle)`` and move that turtle.
        One :class:`Turtle` is created automatically per function.

        Pass the function itself (not a coroutine object) so that a fresh
        coroutine is created on each Start press, enabling clean restarts.
        """
        self._coro_fns = list(coro_fns)
        self._turtles = [Turtle(self) for _ in coro_fns]

    # ------------------------------------------------------------------
    # Widget signal handling (JS → Python via synced traitlets)
    # ------------------------------------------------------------------

    @traitlets.observe("_start_counter")
    def _on_start(self, change) -> None:
        if change["new"] > 0:
            self._start_drawing()

    @traitlets.observe("_stop_requested")
    def _on_stop(self, change) -> None:
        if change["new"]:
            self._stop = True
            self._stop_requested = False  # reset for next press

    # ------------------------------------------------------------------
    # Drawing lifecycle
    # ------------------------------------------------------------------

    def _reset_turtles(self) -> None:
        for t in self._turtles:
            t.segments = []
            t.x = self.width / 2
            t.y = self.height / 2
            t.angle = INITIAL_ANGLE
            t.pen = True
            t.color = Color.CRIMSON.value

    def _start_drawing(self) -> None:
        """Launch the registered drawing coroutines.

        Prefers scheduling an asyncio Task in the running event loop (the
        normal Marimo case, where _on_start fires inside the event loop).
        Falls back to a background thread when called from a non-async
        context (e.g. tests that call _start_drawing directly).
        """
        self._stop = True   # cancel any currently-running drawing
        self._stop = False  # clear immediately for the new run
        self._last_render = 0.0
        self._dirty = False
        self._reset_turtles()

        coros = [fn(self, t) for fn, t in zip(self._coro_fns, self._turtles)]

        async def _run() -> None:
            try:
                if len(coros) == 1:
                    await coros[0]
                else:
                    await asyncio.gather(*coros, return_exceptions=True)
            finally:
                self._flush(show_turtle=False, done=True)

        # Schedule as a concurrent task in Marimo's running event loop.
        # The cell that called set_coroutine() has already returned, so the
        # kernel is free — Start/Stop signals are processed normally.
        asyncio.get_running_loop().create_task(_run())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _flush(self, show_turtle: bool = True, done: bool = False) -> None:
        """Push current state to the JS frontend via the _render traitlet."""
        all_segs = [
            [x1, y1, x2, y2, color]
            for t in self._turtles
            for (x1, y1), (x2, y2), color in t.segments
        ]
        turtle_data = (
            [[t.x, t.y, t.angle] for t in self._turtles] if show_turtle else []
        )
        # Include a monotonic timestamp so the dict is always a new value
        # even when segments haven't changed, ensuring the JS observer fires.
        self._render = {
            "segments": all_segs,
            "turtles": turtle_data,
            "done": done,
            "ts": time.monotonic(),
        }

    def _maybe_render(self) -> None:
        """Rate-limited render: push to JS frontend or call output_fn."""
        now = time.monotonic()
        if self._dirty and (now - self._last_render) >= self.delay:
            if self._output_fn is not None:
                self._output_fn(self._draw())
            else:
                self._flush()
            self._dirty = False
            self._last_render = now

    def _draw(self, show_turtle: bool = True) -> str:
        """Build an SVG string compositing all turtles (used in output_fn / test mode)."""
        lines = ""
        for t in self._turtles:
            for (x1, y1), (x2, y2), color in t.segments:
                lines += (
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                    f'x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{color}" stroke-width="{STROKE_WIDTH}" '
                    f'stroke-linecap="round"/>'
                )
        if show_turtle:
            for t in self._turtles:
                r = math.radians(t.angle)
                pts = " ".join(
                    f"{t.x + TURTLE_RADIUS * math.cos(r + a):.1f},"
                    f"{t.y + TURTLE_RADIUS * math.sin(r + a):.1f}"
                    for a in [0, 2 * math.pi / 3, -2 * math.pi / 3]
                )
                lines += (
                    f'<polygon points="{pts}" fill="{TURTLE_COLOR}"'
                    f' opacity="{TURTLE_OPACITY}"/>'
                )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}"'
            f' height="{self.height}" style="background:{BACKGROUND_COLOR};'
            f'border-radius:{BORDER_RADIUS}px;display:block">'
            f"{lines}</svg>"
        )

    # ------------------------------------------------------------------
    # Direct async run (used by tests)
    # ------------------------------------------------------------------

    async def run(self, *coroutines) -> None:
        """Run coroutines in the current event loop (test / direct-use path).

        Notebook users should use set_coroutine() + the Start button instead.
        """
        self._last_render = 0.0
        try:
            if len(coroutines) == 1:
                await coroutines[0]
            else:
                await asyncio.gather(*coroutines, return_exceptions=True)
        finally:
            if self._output_fn is not None:
                self._output_fn(self._draw(show_turtle=False))
            else:
                self._flush(show_turtle=False, done=True)


class Turtle:
    """
    Async turtle that draws into a World.

    Create via ``World.turtle()`` rather than directly.  Each movement
    method is a coroutine that yields to the event loop after moving so
    that other turtles can run concurrently.
    """

    def __init__(self, world: World):
        self._world = world
        self.x = world.width / 2
        self.y = world.height / 2
        self.angle = INITIAL_ANGLE
        self.pen = True
        self.segments: list = []
        self.color: str = Color.CRIMSON.value

    @property
    def width(self) -> int:
        return self._world.width

    @property
    def height(self) -> int:
        return self._world.height

    def pen_up(self) -> None:
        self.pen = False

    def pen_down(self) -> None:
        self.pen = True

    def goto(self, x: float, y: float) -> None:
        self.x, self.y = x, y

    def set_heading(self, a: float) -> None:
        self.angle = a

    def set_color(self, color: "Color | str") -> None:
        self.color = color.value if isinstance(color, Color) else color

    async def forward(self, dist: float) -> None:
        if self._world._stop:
            return
        r = math.radians(self.angle)
        nx = self.x + dist * math.cos(r)
        ny = self.y + dist * math.sin(r)
        if self.pen:
            self.segments.append(((self.x, self.y), (nx, ny), self.color))
            self.x, self.y = nx, ny
            self._world._dirty = True
            await asyncio.sleep(self._world.delay)
            self._world._maybe_render()
        else:
            self.x, self.y = nx, ny

    async def backward(self, dist: float) -> None:
        await self.forward(-dist)

    def right(self, deg: float) -> None:
        self.angle += deg

    def left(self, deg: float) -> None:
        self.angle -= deg

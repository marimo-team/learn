"""Tests for turtle graphics."""

import asyncio
import math

import pytest

from marimo_learn import Color, Turtle, World

# Use a no-op output function so tests don't need a live Marimo kernel
_NO_OUTPUT = lambda _: None  # noqa: E731

# Short delay to keep tests fast
FAST_DELAY = 0.005


def make_world(**kwargs) -> World:
    """Return a World with rendering disabled."""
    kwargs.setdefault("delay", FAST_DELAY)
    kwargs.setdefault("output_fn", _NO_OUTPUT)
    return World(**kwargs)


# ---------------------------------------------------------------------------
# Color enum
# ---------------------------------------------------------------------------

def test_color_values_are_hex_strings():
    for c in Color:
        assert c.value.startswith("#"), f"{c.name} value {c.value!r} is not a hex string"
        assert len(c.value) == 7, f"{c.name} value {c.value!r} has wrong length"


def test_color_is_string_subclass():
    # Color(str, Enum) means values compare equal to plain strings
    assert Color.CRIMSON == "#e63946"


# ---------------------------------------------------------------------------
# World initialisation
# ---------------------------------------------------------------------------

def test_world_default_dimensions():
    w = make_world()
    assert w.width == 480
    assert w.height == 480


def test_world_custom_dimensions():
    w = make_world(width=200, height=300)
    assert w.width == 200
    assert w.height == 300


def test_world_starts_with_no_turtles():
    w = make_world()
    assert w._turtles == []


def test_world_stop_flag_initially_false():
    w = make_world()
    assert w._stop is False


# ---------------------------------------------------------------------------
# Turtle creation
# ---------------------------------------------------------------------------

def test_turtle_is_turtle_instance():
    w = make_world()
    t = w.turtle()
    assert isinstance(t, Turtle)


def test_turtle_registered_with_world():
    w = make_world()
    t = w.turtle()
    assert t in w._turtles


def test_multiple_turtles_all_registered():
    w = make_world()
    turtles = [w.turtle() for _ in range(3)]
    assert w._turtles == turtles


def test_turtle_starts_at_canvas_centre():
    w = make_world(width=200, height=100)
    t = w.turtle()
    assert t.x == 100.0
    assert t.y == 50.0


# ---------------------------------------------------------------------------
# Turtle movement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_forward_adds_one_segment():
    w = make_world()
    t = w.turtle()
    await t.forward(10)
    assert len(t.segments) == 1


@pytest.mark.asyncio
async def test_segment_format():
    """Each segment is ((x1, y1), (x2, y2), color_str)."""
    w = make_world()
    t = w.turtle()
    x0, y0 = t.x, t.y
    await t.forward(10)
    (x1, y1), (x2, y2), color = t.segments[0]
    assert (x1, y1) == pytest.approx((x0, y0))
    assert color == t.color


@pytest.mark.asyncio
async def test_forward_updates_position():
    w = make_world()
    t = w.turtle()
    t.set_heading(0)  # pointing right
    x0, y0 = t.x, t.y
    await t.forward(50)
    assert t.x == pytest.approx(x0 + 50, abs=1e-6)
    assert t.y == pytest.approx(y0, abs=1e-6)


@pytest.mark.asyncio
async def test_backward_updates_position():
    w = make_world()
    t = w.turtle()
    t.set_heading(0)
    x0, y0 = t.x, t.y
    await t.backward(30)
    assert t.x == pytest.approx(x0 - 30, abs=1e-6)
    assert t.y == pytest.approx(y0, abs=1e-6)


def test_right_increases_angle():
    w = make_world()
    t = w.turtle()
    before = t.angle
    t.right(45)
    assert t.angle == pytest.approx(before + 45)


def test_left_decreases_angle():
    w = make_world()
    t = w.turtle()
    before = t.angle
    t.left(30)
    assert t.angle == pytest.approx(before - 30)


# ---------------------------------------------------------------------------
# Pen up/down
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pen_up_no_segment():
    w = make_world()
    t = w.turtle()
    t.pen_up()
    await t.forward(20)
    assert len(t.segments) == 0


@pytest.mark.asyncio
async def test_pen_up_still_moves():
    w = make_world()
    t = w.turtle()
    t.set_heading(0)
    t.pen_up()
    x0 = t.x
    await t.forward(20)
    assert t.x == pytest.approx(x0 + 20, abs=1e-6)


@pytest.mark.asyncio
async def test_pen_down_after_pen_up_adds_segment():
    w = make_world()
    t = w.turtle()
    t.pen_up()
    await t.forward(10)
    t.pen_down()
    await t.forward(10)
    assert len(t.segments) == 1


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------

def test_set_color_with_enum():
    w = make_world()
    t = w.turtle()
    t.set_color(Color.TEAL)
    assert t.color == Color.TEAL.value


def test_set_color_with_hex_string():
    w = make_world()
    t = w.turtle()
    t.set_color("#abcdef")
    assert t.color == "#abcdef"


@pytest.mark.asyncio
async def test_segment_stores_active_color():
    w = make_world()
    t = w.turtle()
    t.set_color(Color.GOLD)
    await t.forward(10)
    _, _, color = t.segments[0]
    assert color == Color.GOLD.value


# ---------------------------------------------------------------------------
# Stop flag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_flag_prevents_segment():
    """Setting _stop before forward() means no segment is added."""
    w = make_world()
    t = w.turtle()
    w._stop = True
    await t.forward(10)
    assert len(t.segments) == 0


@pytest.mark.asyncio
async def test_stop_flag_prevents_position_change_with_pen_down():
    """With pen down, stop flag returns early before moving."""
    w = make_world()
    t = w.turtle()
    t.set_heading(0)
    x0 = t.x
    w._stop = True
    await t.forward(10)
    assert t.x == pytest.approx(x0)


@pytest.mark.asyncio
async def test_stop_during_gather():
    """A stopper coroutine can interrupt a running multi-step draw via gather."""
    w = make_world(delay=0.01)
    t = w.turtle()

    async def draw_many():
        for _ in range(200):
            await t.forward(1)
            t.right(1.8)

    async def stopper():
        # Let a few steps happen, then signal stop
        await asyncio.sleep(0.05)
        w._stop = True

    await w.run(draw_many(), stopper())

    # Some segments were drawn before the stop
    assert len(t.segments) > 0
    # But drawing stopped well short of 200
    assert len(t.segments) < 200


@pytest.mark.asyncio
async def test_stop_single_coroutine():
    """Stop flag also halts a single-coroutine run (direct await path)."""
    w = make_world(delay=0.01)
    t = w.turtle()

    async def draw_and_stop():
        for _ in range(200):
            await t.forward(1)
            t.right(1.8)

    # Schedule a task that sets _stop after a brief delay
    async def set_stop():
        await asyncio.sleep(0.05)
        w._stop = True

    stop_task = asyncio.create_task(set_stop())
    await w.run(draw_and_stop())
    await stop_task  # ensure cleanup

    assert 0 < len(t.segments) < 200


# ---------------------------------------------------------------------------
# set_coroutine
# ---------------------------------------------------------------------------

def test_set_coroutine_registers_functions():
    async def fn(world, turtle):
        pass

    w = make_world()
    w.set_coroutine(fn)
    assert w._coro_fns == [fn]
    assert len(w._turtles) == 1


def test_set_coroutine_multiple_creates_one_turtle_each():
    async def fn1(world, turtle):
        pass

    async def fn2(world, turtle):
        pass

    w = make_world()
    w.set_coroutine(fn1, fn2)
    assert len(w._coro_fns) == 2
    assert len(w._turtles) == 2


# ---------------------------------------------------------------------------
# _reset_turtles
# ---------------------------------------------------------------------------

def test_reset_turtles_restores_initial_state():
    w = make_world(width=200, height=100)
    t = w.turtle()
    t.x = 0
    t.y = 0
    t.pen_up()
    t.segments = [object()]  # non-empty

    w._reset_turtles()

    assert t.x == pytest.approx(100.0)
    assert t.y == pytest.approx(50.0)
    assert t.pen is True
    assert t.segments == []


# ---------------------------------------------------------------------------
# _on_stop observer
# ---------------------------------------------------------------------------

def test_on_stop_sets_stop_flag():
    w = make_world()
    w._stop_requested = True
    assert w._stop is True
    assert w._stop_requested is False


# ---------------------------------------------------------------------------
# _flush
# ---------------------------------------------------------------------------

def test_flush_populates_render_dict():
    w = make_world()
    w.turtle()
    w._flush()
    assert set(w._render.keys()) == {"segments", "turtles", "done", "ts"}
    assert w._render["done"] is False


def test_flush_done_true():
    w = make_world()
    w._flush(show_turtle=False, done=True)
    assert w._render["done"] is True
    assert w._render["turtles"] == []


# ---------------------------------------------------------------------------
# _maybe_render in widget mode (no output_fn)
# ---------------------------------------------------------------------------

def test_maybe_render_widget_mode_calls_flush():
    w = World(delay=0)
    w.turtle()
    w._dirty = True
    w._last_render = 0.0
    w._maybe_render()
    assert "segments" in w._render


# ---------------------------------------------------------------------------
# run() in widget mode (no output_fn)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_widget_mode_calls_flush():
    w = World(delay=FAST_DELAY)
    t = w.turtle()

    async def draw(world, turtle):
        await turtle.forward(10)

    await w.run(draw(w, t))
    assert w._render.get("done") is True


# ---------------------------------------------------------------------------
# Turtle.width / Turtle.height properties
# ---------------------------------------------------------------------------

def test_turtle_width_property():
    w = make_world(width=320)
    t = w.turtle()
    assert t.width == 320


def test_turtle_height_property():
    w = make_world(height=240)
    t = w.turtle()
    assert t.height == 240


# ---------------------------------------------------------------------------
# Turtle.goto
# ---------------------------------------------------------------------------

def test_goto_sets_position():
    w = make_world()
    t = w.turtle()
    t.goto(12.5, 34.0)
    assert t.x == pytest.approx(12.5)
    assert t.y == pytest.approx(34.0)


# ---------------------------------------------------------------------------
# _on_start observer and _start_drawing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_start_triggers_drawing():
    """Incrementing _start_counter fires _on_start which launches drawing."""
    async def draw(world, turtle):
        await turtle.forward(10)

    w = make_world()
    w.set_coroutine(draw)
    w._start_counter = 1  # fires _on_start -> _start_drawing
    await asyncio.sleep(0.1)  # let the created task run to completion

    assert len(w._turtles[0].segments) == 1

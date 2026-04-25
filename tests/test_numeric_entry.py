"""Unit tests for NumericEntryWidget."""

import pytest
from marimo_learn import NumericEntryWidget

# Named constant matching the widget default
DEFAULT_TOLERANCE = 1e-9


class TestNumericEntryWidget:
    def test_initialization(self):
        w = NumericEntryWidget(question="What is 2 + 2?", correct_answer=4.0)
        assert w.question == "What is 2 + 2?"
        assert w.correct_answer == 4.0
        assert w.tolerance == DEFAULT_TOLERANCE
        assert w.explanation == ""

    def test_custom_tolerance(self):
        w = NumericEntryWidget(question="Pi?", correct_answer=3.14159, tolerance=0.01)
        assert w.tolerance == 0.01

    def test_explanation(self):
        w = NumericEntryWidget(
            question="What is 2 + 2?",
            correct_answer=4.0,
            explanation="Two plus two equals four.",
        )
        assert w.explanation == "Two plus two equals four."

    def test_integer_answer(self):
        w = NumericEntryWidget(question="How many sides does a triangle have?", correct_answer=3)
        assert w.correct_answer == 3.0

    def test_negative_answer(self):
        w = NumericEntryWidget(question="What is -5?", correct_answer=-5.0)
        assert w.correct_answer == -5.0

    def test_zero_answer(self):
        w = NumericEntryWidget(question="What is 0?", correct_answer=0.0)
        assert w.correct_answer == 0.0

    def test_independent_instances(self):
        w1 = NumericEntryWidget(question="Q1", correct_answer=1.0)
        w2 = NumericEntryWidget(question="Q2", correct_answer=2.0)
        assert w1.question != w2.question
        assert w1.correct_answer != w2.correct_answer

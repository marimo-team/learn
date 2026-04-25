"""Unit tests for PredictThenCheckWidget."""

from marimo_learn import PredictThenCheckWidget

QUESTION = "What does this print?"
CODE = "print(1 + 1)"
OUTPUT = "2"
OPTIONS = ["1", "2", "3", "Error"]
CORRECT = 1  # "2" is at index 1


class TestPredictThenCheckWidget:
    def test_initialization(self):
        w = PredictThenCheckWidget(
            question=QUESTION,
            code=CODE,
            output=OUTPUT,
            options=OPTIONS,
            correct_answer=CORRECT,
        )
        assert w.question == QUESTION
        assert w.code == CODE
        assert w.output == OUTPUT
        assert w.options == OPTIONS
        assert w.correct_answer == CORRECT
        assert w.explanations == []
        assert w.explanation == ""

    def test_per_option_explanations(self):
        explanations = ["Too low", "Correct", "Too high", "No error"]
        w = PredictThenCheckWidget(
            question=QUESTION,
            code=CODE,
            output=OUTPUT,
            options=OPTIONS,
            correct_answer=CORRECT,
            explanations=explanations,
        )
        assert w.explanations == explanations

    def test_fallback_explanation(self):
        w = PredictThenCheckWidget(
            question=QUESTION,
            code=CODE,
            output=OUTPUT,
            options=OPTIONS,
            correct_answer=CORRECT,
            explanation="Addition yields 2.",
        )
        assert w.explanation == "Addition yields 2."

    def test_correct_answer_first_option(self):
        w = PredictThenCheckWidget(
            question=QUESTION,
            code=CODE,
            output=OUTPUT,
            options=OPTIONS,
            correct_answer=0,
        )
        assert w.correct_answer == 0

    def test_correct_answer_last_option(self):
        w = PredictThenCheckWidget(
            question=QUESTION,
            code=CODE,
            output=OUTPUT,
            options=OPTIONS,
            correct_answer=len(OPTIONS) - 1,
        )
        assert w.correct_answer == len(OPTIONS) - 1

    def test_independent_instances(self):
        w1 = PredictThenCheckWidget(
            question="Q1", code="x=1", output="", options=["a", "b"], correct_answer=0
        )
        w2 = PredictThenCheckWidget(
            question="Q2", code="x=2", output="", options=["c", "d"], correct_answer=1
        )
        assert w1.question != w2.question
        assert w1.correct_answer != w2.correct_answer

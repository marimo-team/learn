"""Unit tests for MultipleChoiceWidget."""

from marimo_learn import MultipleChoiceWidget


class TestMultipleChoiceWidget:
    """Test suite for MultipleChoiceWidget"""

    def test_initialization(self):
        """Test basic widget initialization"""
        widget = MultipleChoiceWidget(
            question="What is 2 + 2?",
            options=["3", "4", "5"],
            correct_answer=1,
            explanation="Basic math",
        )

        assert widget.question == "What is 2 + 2?"
        assert widget.options == ["3", "4", "5"]
        assert widget.correct_answer == 1
        assert widget.explanation == "Basic math"
        assert widget.value is None

    def test_initialization_without_explanation(self):
        """Test widget initialization without explanation"""
        widget = MultipleChoiceWidget(
            question="What is the capital of France?",
            options=["London", "Paris", "Berlin"],
            correct_answer=1,
        )

        assert widget.question == "What is the capital of France?"
        assert widget.explanation == ""

    def test_options_list(self):
        """Test that options are properly stored as a list"""
        options = ["Option A", "Option B", "Option C", "Option D"]
        widget = MultipleChoiceWidget(
            question="Test?", options=options, correct_answer=2
        )

        assert widget.options == options
        assert len(widget.options) == 4

    def test_correct_answer_index(self):
        """Test correct answer index validation"""
        widget = MultipleChoiceWidget(
            question="Test?", options=["A", "B", "C"], correct_answer=0
        )

        assert widget.correct_answer == 0

        widget2 = MultipleChoiceWidget(
            question="Test?", options=["A", "B", "C"], correct_answer=2
        )

        assert widget2.correct_answer == 2

    def test_value_update(self):
        """Test value property updates"""
        widget = MultipleChoiceWidget(
            question="Test?", options=["A", "B", "C"], correct_answer=1
        )

        # Simulate user selection
        widget.value = {"selected": 1, "correct": True, "answered": True}

        assert widget.value["selected"] == 1
        assert widget.value["correct"] is True
        assert widget.value["answered"] is True

    def test_incorrect_answer(self):
        """Test incorrect answer handling"""
        widget = MultipleChoiceWidget(
            question="Test?", options=["A", "B", "C"], correct_answer=1
        )

        widget.value = {"selected": 0, "correct": False, "answered": True}

        assert widget.value["selected"] == 0
        assert widget.value["correct"] is False

    def test_multiple_widgets(self):
        """Test creating multiple independent widgets"""
        widget1 = MultipleChoiceWidget(
            question="Question 1?", options=["A", "B"], correct_answer=0
        )

        widget2 = MultipleChoiceWidget(
            question="Question 2?", options=["X", "Y"], correct_answer=1
        )

        assert widget1.question != widget2.question
        assert widget1.correct_answer != widget2.correct_answer

    def test_long_question_text(self):
        """Test widget with long question text"""
        long_question = "This is a very long question that spans multiple lines and contains lots of text to test how the widget handles lengthy content."

        widget = MultipleChoiceWidget(
            question=long_question, options=["A", "B"], correct_answer=0
        )

        assert widget.question == long_question

    def test_special_characters_in_options(self):
        """Test options with special characters"""
        widget = MultipleChoiceWidget(
            question="Test?",
            options=["<html>", "x = y & z", "50% off!", "€100"],
            correct_answer=0,
        )

        assert "&" in widget.options[1]
        assert "%" in widget.options[2]
        assert "€" in widget.options[3]

    def test_empty_explanation(self):
        """Test widget with empty explanation"""
        widget = MultipleChoiceWidget(
            question="Test?", options=["A", "B"], correct_answer=0, explanation=""
        )

        assert widget.explanation == ""

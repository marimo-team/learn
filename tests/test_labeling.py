"""Unit tests for LabelingWidget."""

from marimo_learn import LabelingWidget


class TestLabelingWidget:
    """Test suite for LabelingWidget"""

    def test_initialization(self):
        """Test basic widget initialization"""
        widget = LabelingWidget(
            question="Label each line:",
            labels=["Variable", "Function", "Loop"],
            text_lines=["x = 1", "def foo():", "for i in range(10):"],
            correct_labels={0: [0], 1: [1], 2: [2]},
        )

        assert widget.question == "Label each line:"
        assert widget.labels == ["Variable", "Function", "Loop"]
        assert widget.text_lines == ["x = 1", "def foo():", "for i in range(10):"]
        assert widget.correct_labels == {0: [0], 1: [1], 2: [2]}
        assert widget.value is None

    def test_default_lang(self):
        """Test that default language is English"""
        widget = LabelingWidget(
            question="Q",
            labels=["A"],
            text_lines=["line 1"],
            correct_labels={0: [0]},
        )
        assert widget.lang == "en"

    def test_custom_lang(self):
        """Test setting a custom language"""
        widget = LabelingWidget(
            question="Q",
            labels=["A"],
            text_lines=["line 1"],
            correct_labels={0: [0]},
            lang="fr",
        )
        assert widget.lang == "fr"

    def test_multiple_correct_labels_per_line(self):
        """Test that a line can have multiple correct labels"""
        widget = LabelingWidget(
            question="Label the code:",
            labels=["Keyword", "Identifier", "Operator"],
            text_lines=["x += 1"],
            correct_labels={0: [1, 2]},
        )
        assert widget.correct_labels[0] == [1, 2]

    def test_value_update(self):
        """Test updating the value attribute"""
        widget = LabelingWidget(
            question="Label:",
            labels=["A", "B"],
            text_lines=["line 1", "line 2"],
            correct_labels={0: [0], 1: [1]},
        )
        widget.value = {
            "placed_labels": {0: [0]},
            "score": 1,
            "total": 2,
            "correct": False,
        }
        assert widget.value["score"] == 1
        assert widget.value["correct"] is False

    def test_multiple_widgets_independence(self):
        """Test that multiple widgets are independent"""
        widget1 = LabelingWidget(
            question="Q1",
            labels=["A"],
            text_lines=["line 1"],
            correct_labels={0: [0]},
        )
        widget2 = LabelingWidget(
            question="Q2",
            labels=["X"],
            text_lines=["line 2"],
            correct_labels={0: [0]},
        )
        assert widget1.question != widget2.question
        assert widget1.labels != widget2.labels

    def test_special_characters(self):
        """Test with special characters in labels and text"""
        widget = LabelingWidget(
            question="Label the symbols:",
            labels=["α-decay", "β-emission"],
            text_lines=["²³⁸U → ²³⁴Th", "¹⁴C → ¹⁴N"],
            correct_labels={0: [0], 1: [1]},
        )
        assert "α-decay" in widget.labels
        assert "²³⁸U → ²³⁴Th" in widget.text_lines

    def test_correct_labels_structure(self):
        """Test the correct_labels dict structure with various indices"""
        correct = {0: [0, 2], 1: [], 2: [1]}
        widget = LabelingWidget(
            question="Q",
            labels=["X", "Y", "Z"],
            text_lines=["a", "b", "c"],
            correct_labels=correct,
        )
        assert widget.correct_labels[0] == [0, 2]
        assert widget.correct_labels[1] == []
        assert widget.correct_labels[2] == [1]

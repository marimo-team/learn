"""Unit tests for MatchingWidget."""

from marimo_learn import MatchingWidget


class TestMatchingWidget:
    """Test suite for MatchingWidget"""

    def test_initialization(self):
        """Test basic widget initialization"""
        widget = MatchingWidget(
            question="Match the items:",
            left=["A1", "A2", "A3"],
            right=["B1", "B2", "B3"],
            correct_matches={0: 2, 1: 0, 2: 1},
        )

        assert widget.question == "Match the items:"
        assert widget.left == ["A1", "A2", "A3"]
        assert widget.right == ["B1", "B2", "B3"]
        assert widget.correct_matches == {0: 2, 1: 0, 2: 1}
        assert widget.value is None

    def test_equal_column_lengths(self):
        """Test with equal length columns"""
        col_a = ["Item 1", "Item 2", "Item 3", "Item 4"]
        col_b = ["Match A", "Match B", "Match C", "Match D"]

        widget = MatchingWidget(
            question="Test",
            left=col_a,
            right=col_b,
            correct_matches={0: 0, 1: 1, 2: 2, 3: 3},
        )

        assert len(widget.left) == len(widget.right)

    def test_unequal_column_lengths(self):
        """Test with unequal length columns (more options in B)"""
        widget = MatchingWidget(
            question="Match countries to capitals:",
            left=["France", "Japan"],
            right=["Tokyo", "Paris", "Berlin", "Cairo"],
            correct_matches={0: 1, 1: 0},
        )

        assert len(widget.left) == 2
        assert len(widget.right) == 4

    def test_correct_matches_mapping(self):
        """Test correct matches dictionary structure"""
        matches = {0: 2, 1: 0, 2: 1}
        widget = MatchingWidget(
            question="Test",
            left=["A", "B", "C"],
            right=["X", "Y", "Z"],
            correct_matches=matches,
        )

        assert widget.correct_matches[0] == 2
        assert widget.correct_matches[1] == 0
        assert widget.correct_matches[2] == 1

    def test_value_update_partial_matches(self):
        """Test value update with partial matches"""
        widget = MatchingWidget(
            question="Test",
            left=["A", "B"],
            right=["X", "Y"],
            correct_matches={0: 1, 1: 0},
        )

        widget.value = {"matches": {0: 1}, "correct": False, "score": 0, "total": 2}

        assert widget.value["matches"] == {0: 1}
        assert widget.value["correct"] is False

    def test_value_update_all_correct(self):
        """Test value update with all correct matches"""
        widget = MatchingWidget(
            question="Test",
            left=["A", "B", "C"],
            right=["X", "Y", "Z"],
            correct_matches={0: 0, 1: 1, 2: 2},
        )

        widget.value = {
            "matches": {0: 0, 1: 1, 2: 2},
            "correct": True,
            "score": 3,
            "total": 3,
        }

        assert widget.value["correct"] is True
        assert widget.value["score"] == 3

    def test_value_update_all_incorrect(self):
        """Test value update with all incorrect matches"""
        widget = MatchingWidget(
            question="Test",
            left=["A", "B"],
            right=["X", "Y"],
            correct_matches={0: 0, 1: 1},
        )

        widget.value = {
            "matches": {0: 1, 1: 0},
            "correct": False,
            "score": 0,
            "total": 2,
        }

        assert widget.value["score"] == 0
        assert widget.value["correct"] is False

    def test_programming_languages_example(self):
        """Test realistic example with programming languages"""
        widget = MatchingWidget(
            question="Match languages to paradigms:",
            left=["Python", "Haskell", "C"],
            right=["Functional", "Procedural", "Multi-paradigm"],
            correct_matches={0: 2, 1: 0, 2: 1},
        )

        assert "Python" in widget.left
        assert "Functional" in widget.right
        assert widget.correct_matches[0] == 2  # Python -> Multi-paradigm

    def test_geography_example(self):
        """Test realistic example with geography"""
        widget = MatchingWidget(
            question="Match countries to capitals:",
            left=["France", "Japan", "Egypt", "Australia"],
            right=["Cairo", "Paris", "Tokyo", "Canberra"],
            correct_matches={0: 1, 1: 2, 2: 0, 3: 3},
        )

        assert len(widget.left) == 4
        assert widget.correct_matches[0] == 1  # France -> Paris
        assert widget.correct_matches[1] == 2  # Japan -> Tokyo

    def test_special_characters(self):
        """Test items with special characters"""
        widget = MatchingWidget(
            question="Match symbols:",
            left=["α", "β", "γ"],
            right=["alpha", "beta", "gamma"],
            correct_matches={0: 0, 1: 1, 2: 2},
        )

        assert "α" in widget.left
        assert "alpha" in widget.right

    def test_long_text_items(self):
        """Test with long text items"""
        widget = MatchingWidget(
            question="Match descriptions:",
            left=[
                "A very long description that spans multiple words and provides detailed information",
                "Another lengthy explanation",
            ],
            right=["Short match", "Also short"],
            correct_matches={0: 0, 1: 1},
        )

        assert len(widget.left[0]) > 50

    def test_multiple_widgets_independence(self):
        """Test that multiple widgets are independent"""
        widget1 = MatchingWidget(
            question="Q1", left=["A"], right=["B"], correct_matches={0: 0}
        )

        widget2 = MatchingWidget(
            question="Q2", left=["X"], right=["Y"], correct_matches={0: 0}
        )

        assert widget1.question != widget2.question
        assert widget1.left[0] != widget2.left[0]

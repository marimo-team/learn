"""Unit tests for OrderingWidget."""

from marimo_learn import OrderingWidget


class TestOrderingWidget:
    """Test suite for OrderingWidget"""

    def test_initialization_with_shuffle(self):
        """Test basic widget initialization with shuffle enabled"""
        items = ["First", "Second", "Third"]
        widget = OrderingWidget(question="Put in order:", items=items, shuffle=True)

        assert widget.question == "Put in order:"
        assert widget.items == items
        assert widget.shuffle is True
        assert len(widget.current_order) == len(items)
        # Current order should contain all items
        assert set(widget.current_order) == set(items)

    def test_initialization_without_shuffle(self):
        """Test widget initialization with shuffle disabled"""
        items = ["First", "Second", "Third"]
        widget = OrderingWidget(question="Put in order:", items=items, shuffle=False)

        assert widget.shuffle is False
        assert widget.current_order == items

    def test_default_shuffle_true(self):
        """Test that shuffle defaults to True"""
        widget = OrderingWidget(question="Test", items=["A", "B", "C"])

        assert widget.shuffle is True

    def test_items_preserved(self):
        """Test that original items list is preserved"""
        original_items = ["One", "Two", "Three", "Four"]
        widget = OrderingWidget(question="Test", items=original_items, shuffle=True)

        # Original items should not be modified
        assert widget.items == original_items
        # Current order should have all items
        assert sorted(widget.current_order) == sorted(original_items)

    def test_value_update_correct_order(self):
        """Test value update with correct order"""
        items = ["A", "B", "C"]
        widget = OrderingWidget(question="Test", items=items, shuffle=False)

        widget.value = {"order": ["A", "B", "C"], "correct": True}

        assert widget.value["correct"] is True
        assert widget.value["order"] == items

    def test_value_update_incorrect_order(self):
        """Test value update with incorrect order"""
        items = ["A", "B", "C"]
        widget = OrderingWidget(question="Test", items=items, shuffle=False)

        widget.value = {"order": ["C", "A", "B"], "correct": False}

        assert widget.value["correct"] is False
        assert widget.value["order"] != items

    def test_scientific_method_example(self):
        """Test realistic example with scientific method steps"""
        steps = [
            "Ask a question",
            "Do background research",
            "Construct a hypothesis",
            "Test with an experiment",
            "Analyze data",
            "Draw conclusions",
        ]

        widget = OrderingWidget(
            question="Arrange the scientific method steps:", items=steps, shuffle=True
        )

        assert len(widget.items) == 6
        assert "Ask a question" in widget.current_order
        assert set(widget.current_order) == set(steps)

    def test_chronological_events_example(self):
        """Test realistic example with historical events"""
        events = [
            "Declaration of Independence (1776)",
            "Constitution ratified (1788)",
            "Louisiana Purchase (1803)",
            "Civil War begins (1861)",
        ]

        widget = OrderingWidget(
            question="Order these historical events:", items=events, shuffle=True
        )

        assert len(widget.items) == 4
        assert all(event in widget.current_order for event in events)

    def test_numbers_ordering_example(self):
        """Test with numbers as strings"""
        numbers = ["1", "5", "10", "50", "100"]
        widget = OrderingWidget(
            question="Order from smallest to largest:", items=numbers, shuffle=True
        )

        assert widget.items == numbers
        assert set(widget.current_order) == set(numbers)

    def test_empty_items_list(self):
        """Test behavior with empty items list"""
        widget = OrderingWidget(question="Test", items=[], shuffle=True)

        assert widget.items == []
        assert widget.current_order == []

    def test_single_item(self):
        """Test with a single item"""
        widget = OrderingWidget(question="Test", items=["Only item"], shuffle=True)

        assert len(widget.items) == 1
        assert widget.current_order == ["Only item"]

    def test_two_items(self):
        """Test with two items"""
        items = ["First", "Second"]
        widget = OrderingWidget(question="Test", items=items, shuffle=True)

        assert len(widget.items) == 2
        assert set(widget.current_order) == set(items)

    def test_long_text_items(self):
        """Test with long text items"""
        items = [
            "This is a very long item that contains multiple words and should still work fine",
            "Another long item with lots of text to test the widget's handling of lengthy content",
            "A third item that is also quite long",
        ]

        widget = OrderingWidget(question="Test", items=items, shuffle=True)

        assert len(widget.items) == 3
        assert all(len(item) > 30 for item in widget.items)

    def test_special_characters_in_items(self):
        """Test items with special characters"""
        items = [
            "Step 1: Initialize x = 0",
            "Step 2: Check if x < 10",
            "Step 3: Print 'Done!'",
        ]

        widget = OrderingWidget(
            question="Order the code steps:", items=items, shuffle=True
        )

        assert "<" in widget.items[1]
        assert "'" in widget.items[2]

    def test_unicode_characters(self):
        """Test items with unicode characters"""
        items = ["α", "β", "γ", "δ"]
        widget = OrderingWidget(
            question="Order Greek letters:", items=items, shuffle=True
        )

        assert "α" in widget.current_order
        assert "δ" in widget.current_order

    def test_multiple_widgets_independence(self):
        """Test that multiple widgets are independent"""
        widget1 = OrderingWidget(question="Q1", items=["A", "B"], shuffle=False)

        widget2 = OrderingWidget(question="Q2", items=["X", "Y"], shuffle=False)

        assert widget1.items != widget2.items
        assert widget1.question != widget2.question

    def test_shuffle_randomization(self):
        """Test that shuffle actually randomizes (probabilistic test)"""
        items = ["1", "2", "3", "4", "5", "6", "7", "8"]

        # Create multiple widgets and check if at least one is different from original
        orders = []
        for _ in range(10):
            widget = OrderingWidget(question="Test", items=items, shuffle=True)
            orders.append(widget.current_order)

        # At least one should be different from the original (very high probability)
        assert any(order != items for order in orders)

"""Unit tests for FlashcardWidget."""

from marimo_learn import FlashcardWidget

CARDS = [
    {
        "front": "What is a closure?",
        "back": "A function that captures its enclosing scope.",
    },
    {
        "front": "Define memoization.",
        "back": "Caching results of expensive function calls.",
    },
    {"front": "What is recursion?", "back": "A function that calls itself."},
]


class TestFlashcardWidget:
    def test_initialization(self):
        w = FlashcardWidget(question="Study these terms", cards=CARDS)
        assert w.question == "Study these terms"
        assert len(w.cards) == 3
        assert w.shuffle is True
        assert w.value is None

    def test_no_question(self):
        w = FlashcardWidget(cards=CARDS)
        assert w.question == ""

    def test_no_shuffle(self):
        w = FlashcardWidget(cards=CARDS, shuffle=False)
        assert w.shuffle is False

    def test_card_structure(self):
        w = FlashcardWidget(cards=CARDS)
        assert w.cards[0]["front"] == "What is a closure?"
        assert w.cards[0]["back"] == "A function that captures its enclosing scope."
        assert w.cards[2]["front"] == "What is recursion?"

    def test_single_card(self):
        w = FlashcardWidget(cards=[{"front": "F", "back": "B"}])
        assert len(w.cards) == 1

    def test_value_update_in_progress(self):
        w = FlashcardWidget(cards=CARDS)
        w.value = {
            "results": {0: {"rating": "got_it", "attempts": 1}},
            "complete": False,
        }
        assert w.value["results"][0]["rating"] == "got_it"
        assert w.value["results"][0]["attempts"] == 1
        assert w.value["complete"] is False

    def test_value_update_complete(self):
        w = FlashcardWidget(cards=CARDS)
        results = {i: {"rating": "got_it", "attempts": 1} for i in range(3)}
        w.value = {"results": results, "complete": True}
        assert w.value["complete"] is True
        assert len(w.value["results"]) == 3

    def test_multiple_attempts(self):
        w = FlashcardWidget(cards=CARDS)
        w.value = {
            "results": {0: {"rating": "got_it", "attempts": 3}},
            "complete": False,
        }
        assert w.value["results"][0]["attempts"] == 3

    def test_all_rating_types(self):
        w = FlashcardWidget(cards=CARDS)
        w.value = {
            "results": {
                0: {"rating": "got_it", "attempts": 1},
                1: {"rating": "almost", "attempts": 2},
                2: {"rating": "no", "attempts": 1},
            },
            "complete": False,
        }
        assert w.value["results"][1]["rating"] == "almost"
        assert w.value["results"][2]["rating"] == "no"

    def test_independent_instances(self):
        w1 = FlashcardWidget(cards=CARDS[:1], question="Deck 1")
        w2 = FlashcardWidget(cards=CARDS[1:], question="Deck 2")
        assert w1.question != w2.question
        assert len(w1.cards) != len(w2.cards)

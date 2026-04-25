"""Unit tests for ConceptMapWidget."""

from marimo_learn import ConceptMapWidget

CONCEPTS = ["function", "parameter", "argument", "return value"]
TERMS = ["takes", "produces", "is called with"]
CORRECT_EDGES = [
    {"from": "function", "to": "parameter", "label": "takes"},
    {"from": "function", "to": "return value", "label": "produces"},
    {"from": "argument", "to": "parameter", "label": "is called with"},
]


class TestConceptMapWidget:
    def test_initialization(self):
        w = ConceptMapWidget(
            question="Map these:",
            concepts=CONCEPTS,
            terms=TERMS,
            correct_edges=CORRECT_EDGES,
        )
        assert w.question == "Map these:"
        assert w.concepts == CONCEPTS
        assert w.terms == TERMS
        assert w.correct_edges == CORRECT_EDGES
        assert w.value is None

    def test_no_correct_edges(self):
        w = ConceptMapWidget(question="Map:", concepts=CONCEPTS, terms=TERMS)
        assert w.correct_edges == []

    def test_correct_edges_structure(self):
        w = ConceptMapWidget(
            question="Map:", concepts=CONCEPTS, terms=TERMS, correct_edges=CORRECT_EDGES
        )
        assert w.correct_edges[0]["from"] == "function"
        assert w.correct_edges[0]["to"] == "parameter"
        assert w.correct_edges[0]["label"] == "takes"

    def test_two_concepts(self):
        w = ConceptMapWidget(question="Map:", concepts=["A", "B"], terms=["relates to"])
        assert len(w.concepts) == 2

    def test_single_term(self):
        w = ConceptMapWidget(question="Map:", concepts=CONCEPTS, terms=["causes"])
        assert len(w.terms) == 1

    def test_value_update_in_progress(self):
        w = ConceptMapWidget(
            question="Map:", concepts=CONCEPTS, terms=TERMS, correct_edges=CORRECT_EDGES
        )
        w.value = {
            "edges": [{"from": "function", "to": "parameter", "label": "takes"}],
            "score": 0,
            "total": 3,
            "correct": False,
        }
        assert len(w.value["edges"]) == 1
        assert w.value["correct"] is False

    def test_value_update_all_correct(self):
        w = ConceptMapWidget(
            question="Map:", concepts=CONCEPTS, terms=TERMS, correct_edges=CORRECT_EDGES
        )
        w.value = {
            "edges": [
                {"from": e["from"], "to": e["to"], "label": e["label"], "correct": True}
                for e in CORRECT_EDGES
            ],
            "score": 3,
            "total": 3,
            "correct": True,
        }
        assert w.value["correct"] is True
        assert w.value["score"] == 3

    def test_value_update_partial(self):
        w = ConceptMapWidget(
            question="Map:", concepts=CONCEPTS, terms=TERMS, correct_edges=CORRECT_EDGES
        )
        w.value = {"edges": [], "score": 1, "total": 3, "correct": False}
        assert w.value["score"] == 1
        assert w.value["total"] == 3

    def test_independent_instances(self):
        w1 = ConceptMapWidget(question="Q1", concepts=["A", "B"], terms=["x"])
        w2 = ConceptMapWidget(question="Q2", concepts=["C", "D", "E"], terms=["y", "z"])
        assert w1.question != w2.question
        assert len(w1.concepts) != len(w2.concepts)

    def test_many_concepts(self):
        many = [f"concept{i}" for i in range(10)]
        w = ConceptMapWidget(question="Map:", concepts=many, terms=TERMS)
        assert len(w.concepts) == 10

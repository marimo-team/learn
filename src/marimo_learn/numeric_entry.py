"""Numeric Entry Widget for Marimo"""

from pathlib import Path
import traitlets

from .base import BaseWidget

# Default tolerance matches the forma JS default: suitable for exact integer answers
DEFAULT_TOLERANCE = 1e-9


class NumericEntryWidget(BaseWidget):
    """
    A numeric entry question widget.

    Attributes:
        question (str): The question text to display
        correct_answer (float): The expected numeric answer
        tolerance (float): Acceptance window; answer is correct if
            |entered - correct_answer| < tolerance
        explanation (str): Optional explanation shown after answering
        value (dict): Current state with 'entered', 'correct', 'ok',
            and 'answered' keys
    """

    _esm = Path(__file__).parent / "static" / "numeric-entry.js"

    question = traitlets.Unicode("").tag(sync=True)
    correct_answer = traitlets.Float(0.0).tag(sync=True)
    tolerance = traitlets.Float(DEFAULT_TOLERANCE).tag(sync=True)
    explanation = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        question: str,
        correct_answer: float,
        tolerance: float = DEFAULT_TOLERANCE,
        explanation: str = "",
        lang: str = "en",
        **kwargs,
    ):
        super().__init__(
            question=question,
            correct_answer=correct_answer,
            tolerance=tolerance,
            explanation=explanation,
            lang=lang,
            **kwargs,
        )

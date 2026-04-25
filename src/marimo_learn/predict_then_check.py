"""Predict-Then-Check Widget for Marimo"""

from pathlib import Path
import traitlets

from .base import BaseWidget


class PredictThenCheckWidget(BaseWidget):
    """
    A predict-then-check question widget.

    The learner reads a code snippet, selects their predicted output from
    multiple choice options, receives immediate feedback, and can then
    reveal the actual output to verify by running the code themselves.

    Attributes:
        question (str): The question text to display
        code (str): The code block shown to the learner
        output (str): The actual output revealed when the learner clicks
            "Reveal Output"
        options (list): List of candidate output strings
        correct_answer (int): Index of the correct option (0-based)
        explanations (list): Per-option explanation strings shown after
            answering
        explanation (str): Fallback explanation if explanations is omitted
        value (dict): Current state with 'selected', 'correct', and
            'answered' keys
    """

    _esm = Path(__file__).parent / "static" / "predict-then-check.js"

    question = traitlets.Unicode("").tag(sync=True)
    code = traitlets.Unicode("").tag(sync=True)
    output = traitlets.Unicode("").tag(sync=True)
    options = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    correct_answer = traitlets.Int(0).tag(sync=True)
    explanations = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    explanation = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        question: str,
        code: str,
        output: str,
        options: list[str],
        correct_answer: int,
        explanations: list[str] | None = None,
        explanation: str = "",
        lang: str = "en",
        **kwargs,
    ):
        super().__init__(
            question=question,
            code=code,
            output=output,
            options=options,
            correct_answer=correct_answer,
            explanations=explanations or [],
            explanation=explanation,
            lang=lang,
            **kwargs,
        )

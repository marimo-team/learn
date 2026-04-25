"""Flashcard Widget for Marimo"""

from pathlib import Path
import traitlets

from .base import BaseWidget


class FlashcardWidget(BaseWidget):
    """
    A flashcard widget with self-reported spaced repetition.

    Students flip cards to reveal the answer, then rate themselves
    (Got it / Almost / No). Cards rated "Almost" or "No" are re-inserted
    into the queue; the deck is complete when all cards are rated "Got it".

    Attributes:
        question (str): Optional heading shown above the deck
        cards (list): List of dicts with 'front' and 'back' keys
        shuffle (bool): Whether to shuffle the deck initially
        value (dict): State with 'results' (per-card ratings/attempts) and 'complete'
    """

    _esm = Path(__file__).parent / "static" / "flashcard.js"

    question = traitlets.Unicode("").tag(sync=True)
    cards = traitlets.List().tag(sync=True)
    shuffle = traitlets.Bool(True).tag(sync=True)

    def __init__(self, cards, question="", shuffle=True, lang="en", **kwargs):
        super().__init__(cards=cards, question=question, shuffle=shuffle, lang=lang, **kwargs)

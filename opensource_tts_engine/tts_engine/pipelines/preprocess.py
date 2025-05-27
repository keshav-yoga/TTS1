
"""Lightweight text normalization + sentence segmentation."""
import re
from typing import List
import gruut
from ..config import SUPPORTED_LANGS

_whitespace_re = re.compile(r"\s+")

def normalize(text: str) -> str:
    """Basic normalization: collapse whitespace, standardize quotes."""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return _whitespace_re.sub(" ", text).strip()

def segment(text: str, lang: str = "en") -> List[str]:
    """Split into sentences using gruut for language-aware splitting."""
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    sentences = [s.text for s in gruut.sentences(normalize(text), lang=lang)]
    return sentences

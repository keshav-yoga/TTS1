
"""Language-specific grapheme-to-phoneme."""
import functools
from typing import List
try:
    from phonemizer.backend import EspeakBackend
    from phonemizer import phonemize
except ImportError:
    phonemize = None

def g2p(words: List[str], lang: str = "en") -> List[str]:
    if phonemize is None:
        # Fallback to graphemes
        return words
    try:
        return phonemize(words, language=lang, backend='espeak', strip=True, njobs=4).split()
    except Exception:
        return words

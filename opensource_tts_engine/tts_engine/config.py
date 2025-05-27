
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("TTS_MODEL_DIR", str(BASE_DIR / "models"))
CACHE_DIR = os.environ.get("TTS_CACHE_DIR", str(BASE_DIR / "cache"))

SUPPORTED_LANGS = [
    "en", "hi", "te", "bn", "ta", "ml", "kn", "mr",
    "ar", "es", "fr", "ru", "de", "pt", "ko", "ja", "zh"
]

# Default sample rate
SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", 24000))


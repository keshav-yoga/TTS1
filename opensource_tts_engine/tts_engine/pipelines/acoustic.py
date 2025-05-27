
"""Wrapper for multiple acoustic models."""
import os
import torch
from typing import Dict
from enum import Enum

class AcousticModel(str, Enum):
    STYLETTS2 = "styletts2"
    XTTS = "xtts"
    CHATTTS = "chattts"

_model_cache: Dict[AcousticModel, object] = {}

def _load_styletts2():
    from styletts2.inference import StyleTTS2
    return StyleTTS2(device="cuda" if torch.cuda.is_available() else "cpu")

def _load_xtts():
    from TTS.api import TTS
    return TTS(model_name="tts_models/multilingual/xtts_v2")

def _load_chattts():
    from chattts.api import ChatTTS
    return ChatTTS(device="cuda" if torch.cuda.is_available() else "cpu")

_loaders = {
    AcousticModel.STYLETTS2: _load_styletts2,
    AcousticModel.XTTS: _load_xtts,
    AcousticModel.CHATTTS: _load_chattts,
}

def get_acoustic(model: AcousticModel):
    if model not in _model_cache:
        _model_cache[model] = _loaders[model]()
    return _model_cache[model]

def synthesize_mel(text: str, speaker_wav: str = None, lang: str = "en", model: AcousticModel = AcousticModel.XTTS):
    tts = get_acoustic(model)
    if model == AcousticModel.XTTS:
        return tts.tts_to_mel(text, speaker_wav=speaker_wav, language=lang)
    elif model == AcousticModel.STYLETTS2:
        return tts.infer(text)[0]  # mel spectrogram
    elif model == AcousticModel.CHATTTS:
        return tts.tts(text, language=lang, return_mel=True)
    else:
        raise ValueError(f"Unknown acoustic model {model}")

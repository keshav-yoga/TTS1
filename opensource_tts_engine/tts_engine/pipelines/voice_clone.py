
"""Voice cloning via OpenVoice."""
import os, tempfile
from typing import Optional
from openvoice import ToneColorConverter, load_audio
import torch

_model = None

def _load_model():
    global _model
    if _model is None:
        _model = ToneColorConverter('./checkpoints', device='cuda' if torch.cuda.is_available() else 'cpu')
    return _model

def clone(reference_wav: str, target_wav: str, output_path: Optional[str] = None) -> str:
    model = _load_model()
    target, sr = load_audio(target_wav)
    ref, _ = load_audio(reference_wav)
    out = model.convert(target, ref, sr)
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    out.export(output_path, format='wav')
    return output_path

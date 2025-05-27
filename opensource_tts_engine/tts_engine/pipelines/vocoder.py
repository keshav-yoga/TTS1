
"""HiFi-GAN / BigVGAN vocoder wrapper."""
import torch
from typing import Dict
from enum import Enum

class VocoderModel(str, Enum):
    BIGVGAN = "bigvgan"
    HIFIGAN = "hifigan"

_vocoder_cache: Dict[VocoderModel, object] = {}

def _load_bigvgan():
    import bigvgan
    return bigvgan.load_model("nvidia/bigvgan_24khz_100band")

def _load_hifigan():
    from TTS.vocoder.utils.generic_utils import download_model
    from TTS.vocoder.utils.io import load_config
    from TTS.vocoder.models.hifigan import Hifigan
    config_path, model_path, model_item = download_model("vocoder_models/en/ljspeech/hifigan_v2")
    config = load_config(config_path)
    model = Hifigan.init_from_config(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

_loaders = {
    VocoderModel.BIGVGAN: _load_bigvgan,
    VocoderModel.HIFIGAN: _load_hifigan
}

@torch.inference_mode()
def mels_to_audio(mels, model: VocoderModel = VocoderModel.BIGVGAN):
    voc = _vocoder_cache.get(model)
    if voc is None:
        voc = _loaders[model]()
        _vocoder_cache[model] = voc
    if model == VocoderModel.BIGVGAN:
        return voc(mels).cpu().numpy()
    elif model == VocoderModel.HIFIGAN:
        return voc.inference(mels)

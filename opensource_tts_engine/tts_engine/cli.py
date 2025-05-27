
import argparse, soundfile as sf
from pathlib import Path
from .pipelines.acoustic import synthesize_mel, AcousticModel
from .pipelines.vocoder import mels_to_audio
from .pipelines.preprocess import normalize
from .config import SAMPLE_RATE

def main():
    parser = argparse.ArgumentParser(description="OSS TTS Engine CLI")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--out", "-o", default="out.wav")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--acoustic", default=AcousticModel.XTTS, choices=list(AcousticModel))
    args = parser.parse_args()

    mel = synthesize_mel(args.text, lang=args.lang, model=args.acoustic)
    audio = mels_to_audio(mel)
    sf.write(args.out, audio.T, SAMPLE_RATE)
    print(f"Saved speech to {args.out}")

if __name__ == '__main__':
    main()

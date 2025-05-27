
import io, uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from ..pipelines.preprocess import segment
from ..pipelines.acoustic import synthesize_mel, AcousticModel
from ..pipelines.vocoder import mels_to_audio
from ..pipelines.voice_clone import clone as clone_voice

app = FastAPI(title="OpenSourceTTS", version="0.1.0")

@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    lang: str = Form("en"),
    acoustic: AcousticModel = Form(AcousticModel.XTTS),
):
    mel = synthesize_mel(text, lang=lang, model=acoustic)
    audio = mels_to_audio(mel)
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, audio.T, 24000, format='WAV')
    buf.seek(0)
    return StreamingResponse(buf, media_type='audio/wav')

@app.post("/clone")
async def clone_endpoint(
    reference: UploadFile = File(...),
    text: str = Form(...),
    lang: str = Form("en"),
    acoustic: AcousticModel = Form(AcousticModel.XTTS),
):
    ref_path = f"/tmp/{reference.filename}"
    with open(ref_path, 'wb') as f:
        f.write(await reference.read())
    # Generate mel with speaker reference
    mel = synthesize_mel(text, speaker_wav=ref_path, lang=lang, model=acoustic)
    audio = mels_to_audio(mel)
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, audio.T, 24000, format='WAV')
    buf.seek(0)
    return StreamingResponse(buf, media_type='audio/wav')

def run():
    uvicorn.run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    run()

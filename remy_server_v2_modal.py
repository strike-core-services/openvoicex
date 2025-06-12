import modal
from fastapi import HTTPException
from pydantic import BaseModel
import base64
import tempfile

volume = modal.Volume.from_name("openvoice-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")  # git is required here
    .pip_install(
        "torch==2.2.0+cu121",
        "torchaudio==2.2.0+cu121",
        "fastapi[standard]",
        "uvicorn",
        "git+https://github.com/myshell-ai/MeloTTS.git",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .run_commands("python -m unidic download")
    .run_commands("pip install faster-whisper --no-deps")
    .pip_install(
        "av",
        "ctranslate2",
        "whisper-timestamped",
        "wavmark"
    )
    .run_commands("python -m nltk.downloader averaged_perceptron_tagger_eng")
)

app = modal.App("torch-fastapi-app-v2")

class TTSRequest(BaseModel):
    text: str
    speaker_key: str = "EN-BR"
    speed: float = 1.0

@app.function(volumes={"/openvoice-data": volume}, image=image, gpu="T4")
@modal.fastapi_endpoint(method="POST")
def tts_endpoint(req: TTSRequest):
    import torch
    import tempfile
    import base64
    import io
    import os
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    global base_speaker_tts, tone_color_converter, source_se, device, target_se
    from melo.api import TTS

    # Load models only once on first request
    if 'base_speaker_tts' not in globals():
        import sys
        sys.path.append("/openvoice-data")
        from openvoice import se_extractor
        from openvoice.api import  ToneColorConverter

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        ckpt_converter = '/openvoice-data/checkpoints_v2/converter'
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        model = TTS(language='EN', device=device)
        speaker_ids = model.hps.data.spk2id
        reference_speaker = '/openvoice-data/resources/fry_original.mp3'
        target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

    # Run on every request
    try:

        speaker_id = speaker_ids[req.speaker_key]
        speaker_key_formatted = req.speaker_key.lower().replace('_', '-')
        source_se = torch.load(f'/openvoice-data/checkpoints_v2/base_speakers/ses/{speaker_key_formatted}.pth', map_location=device)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as src_wav, \
            tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as out_wav:

            # Generate TTS audio to temp
            model.tts_to_file(req.text, speaker_id, src_wav.name, speed=req.speed)

            # Apply tone conversion
            tone_color_converter.convert(
                audio_src_path=src_wav.name,
                src_se=source_se,
                tgt_se=target_se,
                output_path=out_wav.name,
                message="@MyShell"
            )

            # Read final output and encode to base64
            out_wav.seek(0)
            audio_bytes = out_wav.read()
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return {"success": True,"base64_wav": base64_audio}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
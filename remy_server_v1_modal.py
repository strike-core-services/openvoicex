import modal
from fastapi import HTTPException
from pydantic import BaseModel
import base64
import tempfile

volume = modal.Volume.from_name("openvoice-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .run_commands("pip install faster-whisper --no-deps")
    .pip_install(
        "torch==2.0.0",
        "fastapi[standard]",
        "uvicorn",
        "librosa",
        "pydub",
        "av",
        "ctranslate2",
        "tokenizers==0.15.1",
        "whisper-timestamped",
        "inflect",
        "unidecode",
        "eng_to_ipa",
        "pypinyin",
        "jieba",
        "cn2an",
        "wavmark",
        "numpy==1.26.4"
    )
)

app = modal.App("torch-fastapi-app")

class TTSRequest(BaseModel):
    text: str
    speaker_key: str = "friendly"
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

    # Load models only once on first request
    if 'base_speaker_tts' not in globals():
        import sys
        sys.path.append("/openvoice-data")
        from openvoice import se_extractor
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Import your model classes & utils here
        # from your_module import BaseSpeakerTTS, ToneColorConverter, se_extractor  # Replace with your actual import

        ckpt_base = '/openvoice-data/checkpoints/base_speakers/EN'
        ckpt_converter = '/openvoice-data/checkpoints/converter'

        base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
        base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

        reference_speaker = '/openvoice-data/resources/fry_original.mp3'
        target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

    # Run on every request
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_tts_wav, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_out_wav:

            # 1) Generate TTS to tmp file
            base_speaker_tts.tts(
                text=req.text,
                output_path=tmp_tts_wav.name,
                speaker=req.speaker_key,
                language='English',
                speed=req.speed
            )

            # 2) Run tone color converter from tmp input to tmp output
            tone_color_converter.convert(
                audio_src_path=tmp_tts_wav.name,
                src_se=source_se,
                tgt_se=target_se,
                output_path=tmp_out_wav.name,
                message='@MyShell'
            )

            # 3) Read output and base64 encode
            tmp_out_wav.seek(0)
            audio_bytes = tmp_out_wav.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        return {"success": True, "base64_wav": audio_b64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
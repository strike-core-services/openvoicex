import os
import io
import base64
import torch
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'

# Load once on startup
base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

reference_speaker = 'resources/uncle_original.mp3'
target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

class TTSRequest(BaseModel):
    text: str
    speaker_key: str = "friendly"
    speed: float = 1.0

@app.post("/tts")
def tts_endpoint(req: TTSRequest):
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

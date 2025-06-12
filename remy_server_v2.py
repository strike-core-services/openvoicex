import io
import base64
import torch
import tempfile
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from pydantic import BaseModel
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from pydub import AudioSegment

# Init
app = FastAPI()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load models once
ckpt_converter = 'checkpoints_v2/converter'
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id
reference_speaker = 'resources/uncle_original.mp3'
target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

# Request schema
class TTSRequest(BaseModel):
    text: str
    speaker_key: str = "EN-BR"
    speed: float = 1.0

@app.post("/tts")
def generate_audio(data: TTSRequest):
    speaker_id = speaker_ids[data.speaker_key]
    speaker_key_formatted = data.speaker_key.lower().replace('_', '-')
    source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key_formatted}.pth', map_location=device)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as src_wav, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as out_wav:

        # Generate and convert voice
        model.tts_to_file(data.text, speaker_id, src_wav.name, speed=data.speed)
        tone_color_converter.convert(
            audio_src_path=src_wav.name,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_wav.name,
            message="@MyShell"
        )

        # Load WAV and convert to FLAC
        audio = AudioSegment.from_file(out_wav.name, format="wav")
        flac_io = io.BytesIO()
        audio.export(flac_io, format="flac")
        flac_io.seek(0)

        return StreamingResponse(flac_io, media_type="audio/flac")

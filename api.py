from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import base64
import tempfile
import os

# Import the original Separator class
from audio_separator.separator.separator import Separator

app = FastAPI()

class SeparationRequest(BaseModel):
    audio_file: str
    input_format: str = "wav"
    model: str = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    output_format: str = "WAV"
    output_bitrate: str = None
    normalization_threshold: float = 0.9
    amplification_threshold: float = 0.6
    output_single_stem: str = None
    invert_using_spec: bool = False
    sample_rate: int = 44100
    use_soundfile: bool = False
    mdx_params: dict = Field(default_factory=dict)
    vr_params: dict = Field(default_factory=dict)
    demucs_params: dict = Field(default_factory=dict)
    mdxc_params: dict = Field(default_factory=dict)

# Initialize the Separator
separator = Separator()

@app.post("/separate")
async def separate_audio(request: SeparationRequest):
    try:
        # Decode the base64 audio file
        audio_file_bytes = base64.b64decode(request.audio_file)

        # Create a temporary file to store the decoded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.input_format}") as temp_audio:
            temp_audio.write(audio_file_bytes)
            temp_audio_path = temp_audio.name

        # Update separator with request options
        separator_options = request.dict(exclude={'audio_file', 'input_format'})
        separator.__dict__.update(separator_options)

        # Load the model
        separator.load_model(request.model)

        # Perform separation
        output_files = separator.separate(temp_audio_path)

        # Read the separated files and encode them as base64
        vocal_file_bytes = open(output_files[0], "rb").read()
        background_file_bytes = open(output_files[1], "rb").read()

        vocal_file_base64 = base64.b64encode(vocal_file_bytes).decode("utf-8")
        background_file_base64 = base64.b64encode(background_file_bytes).decode("utf-8")

        return JSONResponse({
            "vocal_file": vocal_file_base64,
            "background_file": background_file_base64
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        os.unlink(temp_audio_path)

@app.get("/models")
async def list_models():
    return separator.list_supported_model_files()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

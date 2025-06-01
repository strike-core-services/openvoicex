# OpenVoice API Server

This repository contains different implementations of the OpenVoice Text-to-Speech (TTS) API server, offering both local and cloud deployment options using FastAPI and Modal.

## Overview

The OpenVoice API server provides text-to-speech capabilities with voice cloning features. It supports:
- Text-to-speech generation with customizable speakers
- Voice tone color conversion
- Multiple deployment options (local and cloud)

## Project Setup

```bash
# Clone this repository
git clone git@github.com:strike-core-services/openvoicex.git

# Download the checkpoints from here and store in the root directory
git clone https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip

# unzip the checkpoints
unzip checkpoints_1226.zip
```

## Server Implementations

### 1. Local FastAPI Server (remy_server_v1.py)

A basic FastAPI implementation for local development and testing.

#### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn remy_server_v1:app --host 0.0.0.0 --port 8000
```

### 2. Modal Cloud Deployment (remy_server_v1_modal.py)

A cloud-ready implementation using Modal.com for scalable deployment.

#### Setup
```bash
# Install Modal CLI
pip install modal

# Deploy to Modal
modal deploy remy_server_v1_modal.py
```

### 3. Enhanced TTS Server (remy_server_v2.py)

An improved version with additional features and optimizations.

#### Setup
Same as the local FastAPI server.

## API Endpoints

### POST /tts

Generates speech from text with customizable parameters.

#### Request Body
```json
{
    "text": "Your text to convert to speech",
    "speaker_key": "friendly",  // Default speaker
    "speed": 1.0               // Speech speed (default: 1.0)
}
```

#### Response
```json
{
    "success": true,
    "base64_wav": "base64_encoded_audio_data"
}
```

## Implementation Details

### Model Loading
- The server loads TTS and tone color converter models on startup
- Models are loaded only once and reused for all requests
- GPU acceleration is automatically used when available

### Audio Processing Pipeline
1. Text-to-Speech Generation
   - Converts input text to speech using the base speaker model
   - Applies specified speaker characteristics

2. Tone Color Conversion
   - Extracts speaker embeddings from reference audio
   - Applies voice characteristics to the generated speech

3. Audio Encoding
   - Converts the processed audio to WAV format
   - Returns base64-encoded audio data

## Configuration

### Local Server
- Models are loaded from local paths
- Supports CPU and GPU inference
- Configurable through environment variables

### Modal Deployment
- Uses Modal volumes for model storage
``` bash
# create a modal volume
modal volume create openvoice-data

# upload the checkpoints folder
modal volume put openvoice-data ./checkpoints

# upload the openvoice folder
modal volume put openvoice-data ./openvoice

# upload the resources folder
modal volume put openvoice-data ./resources
```

- Automatic GPU provisioning
- Scalable deployment with automatic resource management

## Performance Considerations

- Models are loaded once at startup to minimize latency
- Temporary files are used for audio processing
- Automatic cleanup of temporary files
- GPU acceleration when available

## Dependencies

- FastAPI
- PyTorch
- Modal (for cloud deployment)
- OpenVoice TTS checkpoints

## Contributers
- [Shashank] (https://github.com/shashank404error)

Refer to `requirements.txt` for complete dependency list.
# voice-recognition-sber-speech

Client for Sber's SaluteSpeech gRPC API.

## Environment setup

Run the provided script to create a virtual environment and install dependencies:

```bash
bash setup_env.sh
source .venv/bin/activate
```

## Usage

After activating the environment, process an audio file:

```bash
python main.py <path_to_audio>
```

`pydub` requires `ffmpeg` to be installed on your system for non-WAV inputs.

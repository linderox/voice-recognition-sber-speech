# voice-recognition-sber-speech

Client for Sber's SaluteSpeech gRPC API.

## Environment setup

Run the provided script to create a virtual environment and install dependencies:

```bash
bash setup_env.sh
source .venv/bin/activate
```

## Environment variables in Codespaces

The client reads OAuth credentials from environment variables. In a GitHub Codespace you can provide them as repository secrets:

1. Open your repository on GitHub.
2. Navigate to **Settings** → **Secrets and variables** → **Codespaces**.
3. Click **New secret** and add `AUTH_BASIC`, `CLIENT_ID`, and `SCOPE` (optionally `NGW_SKIP_VERIFY`).
4. Rebuild or reopen the Codespace; the variables will be available in the shell. Locally you may instead create a `.env` file with the same keys.

## Testing

After activating the virtual environment, ensure the script compiles:

```bash
python -m py_compile main.py
```

## Usage

After activating the environment, run the script. By default it downloads and
converts the sample m4a file from `http://antonmislavsky.ru/_test-audio.m4a`.
You may optionally pass a local path or another URL:

```bash
python main.py              # uses the default URL
python main.py /path/to.wav # process a local file
python main.py https://host/file.m4a
```

`pydub` requires `ffmpeg` to be installed on your system for non‑WAV inputs.

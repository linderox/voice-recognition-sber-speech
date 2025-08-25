#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "grpcio>=1.62" "protobuf>=4.25" "requests>=2.31" \
            "numpy>=1.24" "soundfile>=0.12" "matplotlib>=3.8" \
            "python-dotenv>=1.0" "pydub>=0.25"

#!/usr/bin/env bash
#
# Set up and run the SELF-HOSTED auto sermon-trimming test on a Raspberry Pi
# (or any Linux box). No cloud API is used.
#
#   ./pi_test.sh path/to/service.mp4      # trim a specific video
#   ./pi_test.sh 2026-04-12               # trim downloaded/<date>/*.mp4
#   SKIP_LLM=1 ./pi_test.sh <input>       # heuristic only (skip the ~1.9GB model)
#
# Stages: ffmpeg audio -> faster-whisper transcript -> hybrid boundary detect
#         -> ffmpeg trim. RAM/CPU/temp are logged per stage (see logs/pi_test.log).
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -lt 1 ]; then
  echo "usage: ./pi_test.sh <video.mp4 | YYYY-MM-DD> [--model tiny|base|small]"
  exit 1
fi

MODEL_FILE="models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

# 1. System packages: ffmpeg (audio/trim) + build tools (llama-cpp-python compiles on ARM)
need_pkg() { dpkg -s "$1" >/dev/null 2>&1 || return 1; }
if ! command -v ffmpeg >/dev/null 2>&1 || ! need_pkg cmake || ! need_pkg build-essential; then
  echo ">> Installing system packages (needs sudo): ffmpeg build-essential cmake python3-venv python3-dev"
  sudo apt-get update
  sudo apt-get install -y ffmpeg build-essential cmake python3-venv python3-dev
fi

# 2. Python virtualenv + dependencies
if [ ! -d .venv ]; then
  echo ">> Creating virtualenv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
echo ">> Installing Python dependencies (llama-cpp-python compiles from source on the Pi; this can take several minutes)"
pip install -r requirements.txt

# 3. Local LLM model (only needed for low-confidence boundaries). Skippable.
if [ "${SKIP_LLM:-0}" != "1" ] && [ ! -f "$MODEL_FILE" ]; then
  echo ">> Downloading local LLM model (~1.9 GB) — one time"
  python - <<'PY'
import os
from huggingface_hub import hf_hub_download
os.makedirs("models", exist_ok=True)
hf_hub_download(repo_id="bartowski/Qwen2.5-3B-Instruct-GGUF",
                filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf", local_dir="models")
print("model ready")
PY
fi

# 4. Run the end-to-end test
echo ">> Running pipeline"
python pi_test.py "$@"

echo ">> Done. Trimmed video + summary are under .asr_test/  (resource log: logs/pi_test.log)"

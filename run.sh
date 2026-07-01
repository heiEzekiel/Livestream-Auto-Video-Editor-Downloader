#!/usr/bin/env bash

# ----------------------------
# Variables Declaration
# ---------------------------- 
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="python"
VENV_DIR="$BASE_DIR/.venv"
REQ_FILE="$BASE_DIR/requirements.txt"

echo "> Base Dir: $BASE_DIR"

# Run from the repo root so all relative paths (config.yaml, downloaded/, models/)
# resolve correctly under cron, where the CWD is otherwise arbitrary.
cd "$BASE_DIR" || { echo "> Cannot cd to $BASE_DIR"; exit 1; }

# ----------------------------
# Single-instance lock
# ----------------------------
# A run (wait-for-live + record the ~2.5h broadcast + processing) can last hours,
# longer than the gap between cron timings. Without this lock, a later cron run
# would start concurrently and its checker could delete the in-progress run's
# files. flock makes only ONE run.sh proceed at a time; overlapping cron timings
# exit immediately and act as genuine retries instead of clobbering each other.
LOCK_FILE="$BASE_DIR/run.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "> Another run is already in progress; exiting."
  exit 0
fi

# ----------------------------
# Python virtual environment dependencies
# ----------------------------
if [ ! -f "$REQ_FILE" ]; then
  echo "> requirements.txt not found at $REQ_FILE"
  exit 1
fi

# ----------------------------
# Python virtual environment
# ----------------------------
if [ ! -d "$VENV_DIR" ]; then
  # NOTE: requirements.txt includes llama-cpp-python, which COMPILES on the Pi.
  # First-time system setup (ffmpeg, build-essential, cmake, python3-dev) is done
  # by pi_test.sh — run that once before the first run.sh on a fresh Pi.
  echo "> Creating virtual environment"
  python3 -m venv "$VENV_DIR"

  # Activate venv
  source $VENV_DIR/bin/activate
  echo "> Using Python: $(which python)"

  # Install dependencies
  echo "> Installing Python dependencies"
  pip install --upgrade pip
  if ! pip install -r "$REQ_FILE"; then
    echo "> Dependency install failed. On a fresh Pi, run ./pi_test.sh once first"
    echo "  (it installs the system build deps llama-cpp-python needs: build-essential, cmake)."
    exit 1
  fi
else
  # Activate venv
  echo "> Using existing virtual environment"
  source $VENV_DIR/bin/activate
  echo "> Using Python: $(which python)"
  echo "> Updating yt-dlp"
  python3 -m pip install -U "yt-dlp[default]"
  python3 -m pip install -U yt-dlp-ejs
fi

# ----------------------------
# Ensure self-hosted ASR model (local LLM used for low-confidence sermon
# boundaries). One-time download; set SKIP_LLM=1 to run heuristic-only.
# Non-fatal: if the download fails, the pipeline still runs heuristic-only.
# (Path is relative to BASE_DIR, which we cd'd into above.)
# ----------------------------
MODEL_FILE="models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
if [ "${SKIP_LLM:-0}" != "1" ] && [ ! -f "$MODEL_FILE" ]; then
  echo "> Downloading local ASR model (one-time, ~1.9GB)"
  if ! python - <<'PY'
import os
from huggingface_hub import hf_hub_download
os.makedirs("models", exist_ok=True)
hf_hub_download(repo_id="bartowski/Qwen2.5-3B-Instruct-GGUF",
                filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf", local_dir="models")
PY
  then
    echo "> Model download failed; continuing heuristic-only (low-confidence boundaries won't use the LLM)."
  fi
fi


# ----------------------------
# 1. Checker -> decide what to do
# ----------------------------
TODAY=$(date +"%Y-%m-%d")

# checker prints exactly one of: TRUE (done) | PROCESS (download exists, just trim)
# | DOWNLOAD (need a fresh download). Logs go to stderr, so $STATUS is the token.
STATUS=$($PYTHON_BIN "$BASE_DIR/checker.py")

if [[ "$STATUS" == *"TRUE"* ]]; then
    echo "> Livestream already processed for $TODAY. Exiting pipeline."
    exit 0
elif [[ "$STATUS" == *"PROCESS"* ]]; then
    echo "> Existing download found for $TODAY; resuming at processing (skipping download)."
else
    # ----------------------------
    # 2. Pre-Setup + Download
    # ----------------------------
    echo "> No usable download for $TODAY. Resolving URL and downloading."
    echo "> Running Pre-Setup"
    URL=$($PYTHON_BIN "$BASE_DIR/pre-setup.py")

    if [[ -z "$URL" ]]; then
      echo "> Livestream url not found"
      exit 0
    fi
    echo "> Completed running | URL:$URL"

    echo "> Starting Livestream download | URL:$URL"
    #$PYTHON_BIN $BASE_DIR/ytarchive-master/ytarchive.py -r 60 $URL  best
    yt-dlp --live-from-start --js-runtimes deno --wait-for-video "60" $URL;

    echo "> Livestream downloaded!";
    sleep 5;
fi

# ----------------------------
# 3. Process Livestream Video
# ----------------------------
echo "> Starting video processing"

FLAG="FALSE"
FLAG=$($PYTHON_BIN "$BASE_DIR/process_vid.py")

if [[ "$FLAG" == *"TRUE"* ]]; then
    echo "> Pipeline completed successfully!"
else
    echo "> Pipeline Failed!"
fi

echo
echo

# livestream-download-and-auto-edit

Automatically download a YouTube livestream at a scheduled time and auto-edit it
to retain only the main segment of the video (e.g. trimming away pre-service and
post-service portions, keeping the core message).

The pipeline scrapes a YouTube channel for the day's livestream, downloads it
with `yt-dlp`, detects where the main message (sermon) starts and ends, and trims
the video to that range with `ffmpeg`.

Boundary detection is **self-hosted** — no audio or transcript leaves the machine
and there is no cloud API or per-call cost, so it runs unattended on a Raspberry
Pi. The **primary** detector transcribes the audio with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and then finds the
sermon with a hybrid content detector: a fast, deterministic heuristic
(handoff-phrase + scripture/announcement lexical scoring) that defers to a small
**local LLM** only for the low-confidence boundaries. See
[Self-hosted ASR (local LLM)](#self-hosted-asr-local-llm) below.

The **fallback** detector uses speaker diarization
([Resemblyzer](https://github.com/resemble-ai/Resemblyzer)): the preacher
reference is *self-enrolled from the audio itself* (the longest continuous speech
run, located via loudness banding) rather than guessed from fixed clock offsets,
and the sermon is found as the long high-confidence **plateau** in the preacher
similarity — anchored on that plateau and extended **asymmetrically**, tightly at
the start (so it never bleeds back into announcements/worship) and forgivingly at
the end (so it keeps the closing/altar-call that follows a quiet stretch). All
thresholds are per-service percentiles, so the trim self-calibrates instead of
depending on a service running to a fixed schedule.

---

## How it works

The pipeline is orchestrated by [`run.sh`](run.sh) and runs in stages:

```text
run.sh  (single-instance flock lock — overlapping cron runs exit instead of clobbering)
 ├─ 1. checker.py      → already done / resume / fresh? (prints TRUE / PROCESS / DOWNLOAD)
 ├─ 2. pre-setup.py    → resolve today's livestream URL (prints URL)
 ├─ 3. yt-dlp          → download the livestream from the start
 └─ 4. process_vid.py  → convert → transcribe → detect sermon span → trim (prints TRUE/FALSE)
```

| Stage | Script | Responsibility |
|-------|--------|----------------|
| **Checker** | [`checker.py`](checker.py) | Decides the pipeline state and prints one of `TRUE` (already downloaded **and** trimmed — nothing to do), `PROCESS` (healthy download exists but no/corrupt trim — resume at processing **without re-downloading**), or `DOWNLOAD` (no usable download — fetch fresh). Only genuinely corrupt/incomplete artifacts are deleted; a healthy download is never removed just because the trim is missing, so an interrupted run resumes safely. |
| **Pre-setup** | [`pre-setup.py`](pre-setup.py) | Confirms today is a Sunday, creates the dated working folders, then scrapes the configured YouTube channel ([`VideoUtils.get_upcoming_streams`](video_utils.py)) for the day's livestream and stores the title/URL in `config.yaml`. |
| **Download** | `yt-dlp` (in `run.sh`) | Downloads the livestream from the start, waiting for it to go live if necessary. |
| **Process** | [`process_vid.py`](process_vid.py) | Renames/moves the download into `downloaded/<date>/`, extracts audio to MP3, then detects the sermon span and trims to `trimmed/<date>/`. **Primary:** self-hosted ASR — transcribe with faster-whisper, then a hybrid heuristic + local-LLM detector ([`asr_local.detect`](asr_local.py), no cloud API). **Fallback:** speaker-diarization similarity plateau ([`AudioUtils.get_trim_range`](audio_utils.py)) if ASR is disabled, errors, or returns an invalid span. Toggle via `asr.enabled` in `config.yaml`. |

### Supporting modules

- [`configs.py`](configs.py) — typed access to `config.yaml` (dot-notation `get`/`update`, plus convenience properties).
- [`audio_utils.py`](audio_utils.py) — audio loading, loudness analysis, MP3 conversion, speaker-segment generation, diarization, and trim detection.
- [`video_utils.py`](video_utils.py) — YouTube channel scraping and `ffmpeg` trimming.
- [`asr_transcribe.py`](asr_transcribe.py) — faster-whisper transcription into a cached JSONL transcript.
- [`asr_local.py`](asr_local.py) — the hybrid sermon detector (heuristic + local-LLM fallback).
- [`asr_llm.py`](asr_llm.py) — the local LLM (`llama-cpp-python`) that refines low-confidence boundaries.
- [`utils.py`](utils.py) — file/date/size helpers shared across stages.
- [`logger.py`](logger.py) — console + file logging (`sermon_pipeline` logger).
- [`resource_monitor.py`](resource_monitor.py) — samples RAM (pipeline + system-wide), CPU, and (on the Pi) SoC temperature/throttling around the heavy stages; warns on low free memory or high temperature.
- [`eval_trim.py`](eval_trim.py) — scores `AudioUtils.get_trim_range` against hand-verified ground truth (`TRUTH`); the regression gate for trim-logic changes.
- [`run_trim_test.py`](run_trim_test.py) — manual harness to diarize/trim real services from `downloaded/` into `.test/` (`--reuse` skips re-diarization).
- [`youtube.py`](youtube.py) / [`telegram.py`](telegram.py) — optional upload / notification helpers (not wired into the main pipeline).

> **Compute footprint (Pi 5 16 GB):** diarization is the heavy stage. On a
> ~2.6 h service it peaks at **~8.7 GB RSS** (it loads the whole audio plus
> embeddings into memory) and ~70–80 % CPU, running ~70–90 s on a fast x86 dev
> box (slower on the Pi). Peak RAM scales with service length, so on 16 GB leave
> headroom for the OS — see [`DESIGN.md`](DESIGN.md) §6 for chunked-embedding and
> systemd cgroup-limit recommendations.

See [`DESIGN.md`](DESIGN.md) for a proposed re-architecture toward a reliable,
resource-aware, unattended Pi deployment.

---

## Prerequisites

- **Python 3.10+**
- **[ffmpeg](https://ffmpeg.org/)** on your `PATH` (used for audio extraction and trimming)
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** on your `PATH` (installed/updated automatically by `run.sh`)

> The pipeline is intended to run on Linux (e.g. a Raspberry Pi via `run.sh`),
> while development happens on Windows. `ffmpeg` and `yt-dlp` must be installed
> on the machine that actually runs the pipeline.

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd livestream-download-and-auto-edit

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) install dev/test dependencies
pip install -r requirements-dev.txt
```

`run.sh` will create the virtual environment and install dependencies for you on
first run, and update `yt-dlp` on subsequent runs. On first run it also downloads
the local ASR model (~1.9 GB) to `models/`; set `SKIP_LLM=1` to skip the LLM and
run the heuristic-only detector. See [Self-hosted ASR (local LLM)](#self-hosted-asr-local-llm).

---

## Configuration

All behaviour is driven by [`config.yaml`](config.yaml). Key sections:

```yaml
metadata:
  service_date: '2026-03-01'   # set automatically each run
  church: 
  title: ''                    # populated by pre-setup.py
  livestream: ''               # auto-refreshed each run with the latest scraped URL
  override_livestream: ''      # set manually to force a specific URL (skips scraping)

# urls (channel URLs) live in config.local.yaml — see "Private config" below.

logging:
  directory: logs
  filename: '{date}_pipeline.log'
  level: DEBUG

path:
  archived: archived
  downloaded: downloaded
  trimmed: trimmed

naming:
  similarity_output: similarity_{date}.txt
  trimmed_video: '{date}_sermon.mp4'
  trimmed_audio: '{date}_sermon.mp3'

audio:
  sample_rate: 48000
  loud_level: -20              # dBFS threshold used to locate the loud (worship) section

asr:                           # self-hosted ASR (primary detector) — see "Self-hosted ASR" below
  enabled: true                # false → skip ASR, use diarization only
  whisper_model: tiny          # faster-whisper size (tiny | base | small)
  start_padding: 5             # seconds backed off the detected start
  end_padding: 5               # seconds added past the detected end

diarization:
  rate: 1.4                    # Resemblyzer embedding rate
  confidence_thresholds:       # legacy reference values; no longer gate the trim
    announcement: 0.8
    preacher: 0.6
  self_enroll:                 # build the preacher reference from the audio itself
    gap_bridge_seconds: 20     # bridge brief pauses when locating the speech run
    min_run_seconds: 300       # ignore candidate speech runs shorter than this
    refine_keep_ratio: 0.5     # 2nd-pass centroid keeps this top-similarity fraction

segment_duration: 256          # length (s) of each context reference clip

speakers:                      # rough start times (minutes) for the context columns
  Pre_svc:      { start: 5 }   # (Preacher is now self-enrolled, not clock-guessed)
  Worship:      { start: 30 }
  Announcement: { start: 50 }
  Preacher:     { start: 70 }

trim_logic:
  smooth_window_seconds: 15    # moving-average window applied before boundary detection
  high_percentile: 65          # the sermon "core" is the longest plateau above this percentile
  low_percentile: 40           # the end extends forward while scores stay above this percentile
  shoulder_ratio: 0.9          # start-extension threshold = low + ratio*(high - low)
  dominance_margin: 0.0        # start gate: require Preacher >= max(Announcement,Worship)+margin
  dominance_sustain_seconds: 45 # dominance must hold this long to mark the sermon start
  min_continuous_seconds: 10   # minimum length of the high-confidence core
  core_gap_seconds: 120        # bridge brief dips when locating the core plateau
  start_gap_seconds: 60        # max dip bridged extending the start back (tight, avoids bleed)
  end_gap_seconds: 300         # max dip bridged extending the end forward (keeps the closing)
  start_padding: 15            # seconds backed off the start so the opening is never clipped
  padding: 8                   # seconds added past the detected end
  end_must_be_after_ratio: 0.9 # legacy; unused by the plateau logic
```

> **Note:** `config.yaml` currently doubles as runtime state — `service_date`,
> `title`, and `livestream` are written back to it on each run.

### Private config (`config.local.yaml`)

Deployment-specific and private values — the channel URLs, and optionally API
keys — live in `config.local.yaml`, which is **git-ignored** so they are never
committed. Values there are merged over `config.yaml` on read (the overlay wins)
and are never written back. Copy the template and fill it in:

```bash
cp config.local.example.yaml config.local.yaml
```

```yaml
# config.local.yaml
urls:
  yyy:     https://www.example.com/yyy
  yyy:     https://www.example.com/yyy
  youtube: https://www.youtube.com/@YourChannel/featured
```

### Credentials

Credentials are kept out of version control (`credentials/` is git-ignored):

- `credentials/yt.json` — Google OAuth client secrets for YouTube upload (used by [`youtube.py`](youtube.py)).
- Telegram bot token — read by [`telegram.py`](telegram.py) (optional notifier).

---

## Self-hosted ASR (local LLM)

Sermon-boundary detection is the **primary** trim method and runs entirely
on-device — no audio or transcript ever leaves the machine, and there is no cloud
API or per-call cost. It has two stages:

1. **Transcription** — [`asr_transcribe.py`](asr_transcribe.py) transcribes the
   service audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (CTranslate2, int8, greedy decode) and caches the result as a JSONL transcript
   under `transcripts/`, so re-runs are free.
2. **Detection** — [`asr_local.py`](asr_local.py) finds the sermon span with a
   hybrid detector:
   - a fast, deterministic **heuristic** (`heuristic_detect`) scores each segment
     for handoff phrases ("welcome Pastor …", "the preaching of the word") and
     sermon-vs-announcement vocabulary, returning boundaries **with a confidence**;
   - for any **low-confidence** boundary (e.g. a service with no spoken handoff),
     it defers to a small **local LLM** ([`asr_llm.py`](asr_llm.py) — Qwen2.5-3B
     Instruct, `Q4_K_M` GGUF via `llama-cpp-python`). To keep it cheap on a Pi the
     model is sent **only a short window centered on the uncertain boundary**; the
     confident boundary stays the heuristic anchor and is never sent.

If ASR is disabled (`asr.enabled: false`), errors, or returns an invalid span,
the pipeline automatically falls back to the diarization detector.

### The local model

`run.sh` downloads the GGUF model once to `models/` (git-ignored, ~1.9 GB) on the
first run. The download is **non-fatal**: if it fails, the pipeline still runs the
heuristic-only path. faster-whisper downloads its own (small) model automatically
on first transcription.

`llama-cpp-python` has no prebuilt aarch64 wheel, so it **compiles** on the Pi
during `pip install`. Install the build tools once (see [`pi_test.sh`](pi_test.sh)):

```bash
sudo apt-get install -y build-essential cmake ffmpeg python3-dev
```

### Tuning (environment variables)

| Variable | Default | Effect |
| --- | --- | --- |
| `SKIP_LLM` | `0` | `1` = heuristic-only; the local LLM is never loaded. |
| `ASR_LLM_MODEL` | `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` | Path to the GGUF model. |
| `ASR_REFINE_WINDOW_S` | `420` | ± window (seconds) sent to the LLM around an uncertain boundary. |
| `ASR_LLM_NCTX` | `20480` | LLM context size (KV-cache allocation). |
| `ASR_LLM_THREADS` | all cores | CPU threads for the LLM (lower it to trade speed for a cooler Pi). |

Transcribe a service by hand (writes a cached JSONL under `transcripts/`):

```bash
python asr_transcribe.py 2026-03-01 --model tiny
```

---

## Usage

Run the full pipeline:

```bash
./run.sh
```

This is typically scheduled (e.g. via `cron`) to run shortly before the weekly
livestream. The script exits early if today's stream was already processed.

### Running stages individually

```bash
python checker.py                 # prints TRUE (done) / PROCESS (resume) / DOWNLOAD (fetch)
python pre-setup.py               # prints the resolved livestream URL
python asr_transcribe.py <date>   # (optional) pre-transcribe a service to cached JSONL
python process_vid.py             # processes the most recent download, prints TRUE on success
```

### Output layout

```text
downloaded/<date>/   # raw download + extracted MP3 + transcript (.jsonl) + similarity_<date>.txt
trimmed/<date>/      # <date>_sermon.mp4  (the auto-edited result)
archived/<date>/     # archive folder
transcripts/<date>.jsonl  # cached transcript when using asr_transcribe.py directly
logs/<date>_pipeline.log
```

---

## Testing

Unit tests live in [`tests/`](tests/) and cover the pure logic (config handling,
trim detection, file/date helpers, channel scraping, and ffmpeg command
construction). Network and `ffmpeg` calls are mocked, so the suite runs offline.

```bash
pip install -r requirements-dev.txt
python -m pytest          # or: python -m pytest -v
```

---

## Project structure

```text
.
├── run.sh                     # pipeline orchestrator (Linux, single-instance lock)
├── checker.py                 # stage 1: 3-state guard (TRUE/PROCESS/DOWNLOAD)
├── pre-setup.py               # stage 2: resolve livestream URL
├── process_vid.py             # stage 4: convert, detect sermon span, trim
├── asr_transcribe.py          # ASR: faster-whisper transcription (cached JSONL)
├── asr_local.py               # ASR: hybrid heuristic + local-LLM detector
├── asr_llm.py                 # ASR: local LLM boundary refinement (llama-cpp)
├── audio_utils.py             # fallback: audio + diarization + trim detection
├── video_utils.py             # YouTube scraping + ffmpeg trimming
├── resource_monitor.py        # RAM/CPU/temp monitoring around heavy stages
├── utils.py                   # shared file/date/size helpers
├── logger.py                  # logging setup
├── configs.py                 # config accessor (config.yaml + config.local.yaml overlay)
├── config.yaml                # configuration + runtime state
├── config.local.example.yaml  # template for the git-ignored private overlay
├── youtube.py                 # optional: YouTube upload helper
├── telegram.py                # optional: Telegram notifier
├── tests/                     # pytest suite
├── requirements.txt           # runtime dependencies
└── requirements-dev.txt       # test dependencies
```

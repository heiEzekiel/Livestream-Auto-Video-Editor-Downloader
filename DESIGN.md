# Re-architecture proposal

Goal (from [README.md](README.md)): *automatically download the weekly YouTube
livestream and auto-edit it down to just the sermon* — reliably, unattended, on
a **Raspberry Pi 5 (16 GB)** that is also running a normal OS and other
processes.

This document proposes how to evolve the current scripts toward that goal. It is
a menu, not a mandate — each section stands alone and is ordered roughly by
value-to-effort.

---

## 1. Where we are today

```
run.sh  (bash glue, parses TRUE/FALSE from stdout)
 ├─ checker.py     prints TRUE/FALSE
 ├─ pre-setup.py   resolves URL, writes it into config.yaml
 ├─ yt-dlp         downloads
 └─ process_vid.py convert → diarize → trim
```

Pain points:

- **`config.yaml` doubles as runtime state.** `service_date`, `title`,
  `livestream` are written back each run ([configs.py](configs.py),
  [process_vid.py](process_vid.py)). Config (committed, static) and state
  (mutable, per-run) have opposite lifecycles; mixing them makes runs
  non-reproducible and races likely if anything overlaps.
- **Orchestration is stringly-typed.** Stages communicate by printing
  `TRUE`/`FALSE` and by side effects on the filesystem. There is no single
  source of truth for "what happened in this run", no resume, no structured
  errors.
- **One trim strategy, hard-wired.** The detector lives inside
  [audio_utils.py](audio_utils.py); there is no seam to swap or ensemble
  strategies, which makes the accuracy work we just did hard to extend.
- **Not resource-aware.** Diarization loads the whole service into RAM and is
  CPU/thermal-heavy; nothing prevents it from overlapping the download or
  fighting the OS for memory on the Pi. (Addressed in part by the new
  [resource_monitor.py](resource_monitor.py).)
- **Accuracy has a ceiling.** Trim is driven purely by speaker-embedding
  similarity; ~15/19 services are within ~20 s but a handful are minutes off,
  and acoustic features alone cannot resolve them (see §5).

---

## 2. Target shape

A single Python entrypoint orchestrating explicit, idempotent stages over a
small per-run **state record**, with a pluggable detector and resource guards:

```
sermon_pipeline/
  __main__.py        # `python -m sermon_pipeline` — the only entrypoint
  state.py           # JobState (sqlite or per-run JSON), separate from config
  stages/
    resolve.py       # find today's stream  (was pre-setup.py)
    download.py      # yt-dlp wrapper, retries, partial-file guard
    audio.py         # mp4 -> mp3
    detect.py        # SermonDetector interface + implementations
    trim.py          # ffmpeg
  detectors/
    similarity.py    # current self-enrolled + plateau logic
    ...              # future strategies (see §5)
  resource_monitor.py
  eval/              # eval_trim.py + ground-truth fixtures (regression gate)
```

`run.sh` shrinks to "ensure venv, update yt-dlp, `python -m sermon_pipeline`".

---

## 3. Separate config from state  *(high value, low effort)*

- Keep [config.yaml](config.yaml) **read-only** (tuning + paths only).
- Move per-run fields (`service_date`, `title`, `livestream`, stage status,
  detected timestamps, resource metrics) into a **state store**: either
  `state/<date>.json` or a tiny SQLite `runs` table.
- Benefits: reproducible runs, trivial resume, no write-back races, and the
  state record becomes the natural place to record what the detector decided
  (great for the eval harness and debugging).

---

## 4. Explicit, resumable orchestration  *(high value, medium effort)*

- Model the pipeline as ordered stages with typed results, each writing its
  outcome to the state record and **skipping if already done** (idempotent).
- Replace `print("TRUE")` / stdout parsing with exit codes + the state record.
- Add per-stage retries where it matters (network/`yt-dlp`), and a single
  structured log line per stage (stage, status, duration, key outputs).
- The existing health checks in [checker.py](checker.py) (size thresholds,
  trimmed-smaller-than-source, single source mp4) become a `verify` stage
  rather than a separate script.

---

## 5. Make the detector pluggable, and plan the accuracy ceiling

Define one interface and keep today's logic as the first implementation:

```python
class SermonDetector(Protocol):
    def detect(self, audio: Path, ctx: RunContext) -> TrimRange: ...
```

- `SimilarityDetector` — the current self-enrolled embedding + plateau +
  asymmetric extension + dominance gate (in [audio_utils.py](audio_utils.py)).
- This seam lets us **ensemble** or **fall back** between strategies and A/B
  them through `eval_trim.py` without touching the pipeline.

**Why a new strategy is needed for the last cases.** We validated that a
Pi-viable music/speech layer does *not* help: spectral flatness, pulse/PLP, and
4 Hz modulation do not separate worship from preaching in these recordings
(congregational singing is voice-forward and the sermons sit over music beds),
and HPSS/percussive-ratio is far too slow for the Pi. The residual errors are
**semantic, not acoustic** (e.g. one service's keep-out block scores *higher*
on the preacher than another service's keep-in block). The only signal that
distinguishes them is *content*:

- **ASR-based detection (recommended next lever).** Transcribe with
  `whisper.cpp` (`tiny`/`base` int8) — runs on the Pi 5 CPU, no Python ML stack,
  small footprint. Then find the sermon by content: the long monologue, a
  scripture reading, an opening like "let's open our Bibles", a sustained single
  speaker. Text is robust where embeddings are ambiguous. Cost: ~minutes of CPU
  for `tiny` on a 2.5 h service; run it once, cache the transcript in the state
  record. Use it to *refine* the similarity boundaries, not replace them.
- Keep `SimilarityDetector` as the fast default; invoke ASR only to disambiguate
  low-confidence boundaries (when the start is far from the high-confidence
  plateau, or dominance is marginal).

---

## 6. Resource-aware execution on the Pi  *(directly addresses the new ask)*

- [resource_monitor.py](resource_monitor.py) now wraps the heavy stages
  (convert / diarize / trim) and logs pipeline RSS, **system-wide** memory
  pressure (OS + other processes), CPU, and SoC temperature; it warns on low
  free RAM (<1 GB) and thermal throttling (>80 °C).
- Use it to **gate** as well as observe:
  - Don't start diarization while a download is still running (serialize the
    two heaviest stages); check `snapshot()["available_mb"]` before the big
    `librosa.load`.
  - Cap CPU threads for `torch`/`librosa` on the Pi (e.g. `OMP_NUM_THREADS`,
    `torch.set_num_threads`) to leave headroom for the OS and avoid thermal
    spikes — expose as config.
  - Process audio at 16 kHz mono (already the diarization rate) and consider
    **streaming/chunked** embedding so peak RAM is bounded regardless of service
    length (today the whole wav + embeddings sit in memory).
- Replace `cron` + `run.sh` with a **systemd timer + service**: gives logging,
  automatic restart, `MemoryMax=`/`CPUQuota=` cgroup limits, and `Nice=`/
  `IOSchedulingClass=` so the pipeline yields to interactive use.

---

## 7. Observability & regression safety  *(low effort, compounding value)*

- Keep `eval_trim.py` + the validated `TRUTH` set as a **regression gate**: run
  it in CI (or a pre-deploy check) so a tuning change that helps one service and
  silently breaks three is caught — exactly the failure mode we hit by hand.
- Emit structured logs (one JSON line per stage) and persist the resource
  summary into the state record for trend analysis on the Pi.
- Store per-run the detector's intermediate evidence (core span, thresholds,
  dominance decision) so future debugging doesn't require re-running diarization.

---

## 8. Robustness & housekeeping

- `yt-dlp`: retries/backoff, resume partial downloads, verify duration/size
  before processing; surface "stream not live yet" vs "failed" distinctly.
- Atomic file moves (write to temp, rename) and a disk-space precheck (a 2.5 h
  1080p mp4 is ~1.2 GB; the Pi's storage fills fast).
- Retention/archive policy for `downloaded/`, `trimmed/`, `archived/` so the Pi
  doesn't run out of space unattended.
- Packaging: a `pyproject.toml` with pinned deps and a CPU-only `torch` wheel
  (much smaller on the Pi); make `matplotlib` (plotting) an optional extra.

---

## 8a. ASR content layer (prototype — implemented)

The acoustic similarity detector plateaued at ~15/19 starts within 20 s; the
residual failures are **semantic**, not acoustic (pre-sermon worship/testimony
scores like the preacher; some preacher-led segments aren't the message). We
validated that lightweight music/speech features cannot separate these in these
recordings, and built the content-based layer instead.

**Pipeline (per service):**

```text
mp4 ──ffmpeg──▶ mp3 ──faster-whisper(tiny,int8,CPU)──▶ transcripts/<date>.jsonl
                                                            │  (timestamped segments, cached)
                                                            ▼
                                   Claude (Opus 4.8): identify sermon span
                                                            │  structured output + adaptive thinking
                                                            ▼
                                   Claude: adversarial verify (try to move each boundary)
                                                            ▼
                          .asr_test/<date>/ : boundary.json + summary.txt + <date>_sermon.mp4
```

**Components:**
- [`asr_transcribe.py`](asr_transcribe.py) — audio → cached JSONL transcript (faster-whisper, CPU, streams audio → bounded RAM; wrapped in `ResourceMonitor`).
- [`asr_detect.py`](asr_detect.py) — transcript → sermon span via the Claude API (`output_config.format` structured output, adaptive thinking, streamed), then an **adversarial verification** call that tries to move each boundary. Decoupled `detect(path, client)` for testing.
- [`asr_results.py`](asr_results.py) — writes `.asr_test/<date>/` records and trims the video, applying a configurable end padding (`END_PADDING_S = 5`).
- [`asr_eval.py`](asr_eval.py) — scores detected vs the validated `TRUTH` benchmark (shared with `eval_trim.py`).
- In-environment validation uses a Workflow ([asr_validate.workflow.js](asr_validate.workflow.js)) because the SDK key isn't present in the tool sandbox; production uses `asr_detect.py` directly on the Pi where `ANTHROPIC_API_KEY` is set.

**Key boundary rules (learned from user-validated cases):**
- START = where the service is **handed to the preacher** (host welcome / preacher takes the stage), *including* preacher-led scripture recitation/ministry before the teaching — not where expository teaching begins. (Fixed 04-26, which was +933 s when anchored on teaching.)
- END = through the preacher-led closing prayer/altar call, **before** post-service announcements / final song set / dismissal, then **+5 s** padding.
- Residual: a worship song that segues directly into the message (no spoken handoff) is ~40 s finer than the transcript reliably resolves; such starts may need a small manual nudge.

**Cost/footprint:** transcription is the only heavy local step (CPU-only, ~3–5.5 GB RAM, ~18–24× realtime on x86; GPU unused — none on the Pi). Detection is 2 Claude calls per service on a weekly job.

## 8b. How this differs from the original design

| Aspect | Original (pre-ASR) | After self-enrollment + percentile trim | After ASR content layer |
|---|---|---|---|
| Preacher reference | **Clock-guessed** 256 s clip at a fixed minute | **Self-enrolled** from the longest speech run | (unchanged; ASR is independent of it) |
| Trim signal | Absolute thresholds (0.8/0.6) on role similarity | **Relative percentiles + plateau + asymmetric extension + dominance gate** | **Transcript content** read by an LLM |
| Start accuracy | brittle; minutes off when service ran off-schedule | 15/19 within ~20 s; ~4 semantic outliers (up to +17 min) | semantic outliers fixed to seconds (e.g. +1011 s → −5 s) |
| Failure mode | wrong clock guess → garbage scores | pre-sermon worship scores like the preacher (acoustic ambiguity) | (resolved via content; new residual is fine-grained worship-lead-in framing) |
| Heavy compute | Resemblyzer embed (~8.7 GB peak, CPU) | same | + faster-whisper transcription (~3–5.5 GB, CPU) |
| External dependency | none (all local) | none | one Claude API call/week (transcription stays local) |
| New tooling | — | `eval_trim.py` regression gate, `resource_monitor.py` | `asr_*`, `.asr_test/`, validation workflow |

The original design assumed services ran to a **fixed schedule** (clock-guessed references, fixed time windows). Each layer removed an assumption: self-enrollment removed "the preacher is at minute 70"; percentile/plateau removed "0.6/0.8 are universal thresholds"; the ASR layer removes "acoustics can tell sermon from worship/announcements" — replacing it with what a human actually uses to tell them apart: the words.

## 8c. Self-hosted hybrid detector (no cloud API)

To remove the Claude API dependency (cost, privacy, offline-on-Pi), the content
detector was reimplemented to run **entirely on-device** as a hybrid:

```text
transcript ─▶ heuristic_detect (asr_local.py)  ── high confidence ──▶ boundary
                       │  (handoff phrases + lexical scoring)
                       └─ low confidence ─▶ local LLM (asr_llm.py)  ──▶ boundary
                                            Qwen2.5-3B Q4 via llama-cpp-python (CPU)
```

- [`asr_local.py`](asr_local.py) — deterministic, zero new deps. START = the last
  genuine preacher-handoff phrase ("welcome Pastor X", "preaching of the word",
  excluding "...for the announcements"); END = last preacher-led closing cue
  (altar call / benediction) before a dismissal cue. Emits confidences.
- [`asr_llm.py`](asr_llm.py) — runs only for low-confidence boundaries (no spoken
  handoff, multi-sermon). A small quantized instruct model reads the transcript
  down-sampled to ~20 s lines and returns the boundary; the confident heuristic
  side is kept as an anchor.

**Validation (9 services vs ground truth):** the heuristic alone nails the 6
clean-handoff services (mean 9 s, all ≤20 s). The 3 no-handoff/two-sermon
services drop to the local LLM. Excluding the irreducible two-sermon 04-05,
the hybrid is **mean 24 s, 7/8 within 60 s** — matching the Claude-API version,
fully offline. 04-05 (two distinct preaching segments; user wants the second)
is unresolved by any automatic method.

**Footprint:**

- heuristic: pure Python, ~0 RAM, milliseconds — handles ~2/3 of services with no model at all.
- local LLM: ~1.9 GB model + ~3–4 GB RAM during inference; minutes per service on the Pi 5 CPU, invoked for only ~1/3 of services. GPU unused (Pi has none).
- transcription (faster-whisper, unchanged): CPU-only, ~3–5.5 GB, ~5–9 min/service.

**Note on this validation:** the heuristic was run on-device; the local-LLM step
was validated on the exact down-sampled input it receives (the dev laptop's
i9-10980HK lacks AVX-512, so the prebuilt llama.cpp wheel can't load the model
there — it runs on the Pi's ARM build or any AVX2 host). `pip install
llama-cpp-python` builds the right binary on the Pi.

## 9. Suggested order

1. **§3 config/state split** and **§7 eval-as-gate** — cheap, unlock everything else.
2. **§6 resource gating** + systemd — needed for reliable unattended Pi runs.
3. **§4 orchestration** refactor into stages.
4. **§5 detector interface**, then prototype the **ASR refinement** for the
   semantic edge cases.

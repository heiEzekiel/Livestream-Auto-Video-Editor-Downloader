"""
ASR transcription stage (DESIGN.md §5).

Transcribes a service's audio into a cached, timestamped transcript that the
content-based detector (asr_detect.py) reasons over. Uses faster-whisper
(CTranslate2 int8) which streams audio (low, bounded RAM) and runs on the
Raspberry Pi 5 CPU as well as x86; production may instead use whisper.cpp.

Transcripts are cached as JSONL (one segment per line, absolute seconds) so
re-runs are free and the heavy ASR step happens once per service.

CLI:
    python asr_transcribe.py 2026-01-04 [2026-04-12 ...] [--model tiny]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from logger import setup_logger
from configs import Configs
from resource_monitor import ResourceMonitor, log_environment

cfg = Configs(Path("./config.yaml"))
logger = setup_logger(Path("logs/asr.log"), cfg.log_level)

DOWNLOADED = Path("./downloaded")
TRANSCRIPTS = Path("./transcripts")


def find_mp3(date: str) -> Path | None:
    folder = DOWNLOADED / date
    pref = folder / f"{date}.mp3"
    if pref.exists():
        return pref
    others = list(folder.glob("*.mp3"))
    return others[0] if others else None


def transcribe(mp3: Path, out_path: Path, model_size: str = "tiny") -> Path:
    """Transcribe ``mp3`` to a JSONL transcript at ``out_path`` (cached)."""
    if out_path.exists():
        logger.info(f"Transcript exists, skipping: {out_path}")
        return out_path

    from faster_whisper import WhisperModel  # imported lazily (heavy)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 4)

    t0 = time.time()
    # beam_size=1 (greedy): the default beam_size=5 decodes 5 hypotheses for a
    # boundary-detection transcript that only needs ~20s-bin wording — greedy is a
    # free ~20-40% speedup on the Pi CPU with no effect on handoff/lexical scoring.
    segments, info = model.transcribe(
        str(mp3), language="en", beam_size=1, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    n = 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for seg in segments:  # generator -> work happens here
            f.write(json.dumps({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }) + "\n")
            n += 1
    tmp.replace(out_path)  # atomic

    dur = time.time() - t0
    audio_len = getattr(info, "duration", 0) or 0
    rtf = (audio_len / dur) if dur else 0
    logger.info(f"Transcribed {mp3.name}: {n} segments, {audio_len:.0f}s audio "
                f"in {dur:.0f}s ({rtf:.1f}x realtime, model={model_size}) -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Transcribe service audio (cached).")
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--model", default="tiny", help="faster-whisper model size")
    args = ap.parse_args()

    log_environment()
    for date in args.dates:
        mp3 = find_mp3(date)
        if mp3 is None:
            logger.error(f"[{date}] no mp3 found under {DOWNLOADED/date}")
            continue
        out = TRANSCRIPTS / f"{date}.jsonl"
        with ResourceMonitor(f"asr:{date}"):
            transcribe(mp3, out, args.model)


if __name__ == "__main__":
    main()

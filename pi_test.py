#!/usr/bin/env python3
"""
End-to-end test of the SELF-HOSTED auto sermon-trimming pipeline (Pi-runnable).

Runs the whole local chain on one video and writes the trimmed result, logging
RAM / CPU / GPU(if any) / temperature for each heavy stage so you can see how it
behaves on the Pi:

    mp4 -> (ffmpeg) mp3 -> (faster-whisper) transcript
        -> (hybrid heuristic + local LLM) sermon span -> (ffmpeg) trimmed mp4

No cloud API is used. The local LLM only runs for low-confidence boundaries; if
its model isn't present, the heuristic result is used and a note is logged.

Usage:
    python pi_test.py path/to/service.mp4         # a specific file
    python pi_test.py 2026-04-12                  # finds downloaded/<date>/*.mp4
    python pi_test.py <input> --model tiny        # whisper size (tiny|base|small)

Output: .asr_test/<name>/<name>_sermon.mp4  + summary.txt + boundary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from configs import Configs
from logger import setup_logger
from audio_utils import AudioUtils
from video_utils import VideoUtils
from resource_monitor import ResourceMonitor, log_environment
import asr_transcribe
import asr_local
from asr_results import END_PADDING_S

cfg = Configs(Path("./config.yaml"))
logger = setup_logger(Path("logs/pi_test.log"), cfg.log_level)
a_utils = AudioUtils()
v_utils = VideoUtils()


def _hms(x):
    x = int(x)
    return f"{x // 3600}:{(x % 3600) // 60:02d}:{x % 60:02d}"


def resolve_input(arg: str) -> tuple[Path, str]:
    """Return (mp4_path, name) from a file path or a YYYY-MM-DD date folder."""
    p = Path(arg)
    if p.suffix.lower() == ".mp4":
        if not p.exists():
            raise SystemExit(f"file not found: {p}")
        return p, p.stem.split()[0] if p.stem else p.stem
    folder = Path("downloaded") / arg
    if folder.is_dir():
        from run_trim_test import find_source_mp4
        mp4 = find_source_mp4(folder)
        if mp4:
            return mp4, arg
    raise SystemExit(f"no .mp4 found for '{arg}' (pass a file path or a downloaded/<date> folder)")


def main():
    ap = argparse.ArgumentParser(description="Self-hosted auto sermon-trim test.")
    ap.add_argument("input", help="path to an .mp4, or a YYYY-MM-DD date in downloaded/")
    ap.add_argument("--model", default="tiny", help="faster-whisper model size")
    args = ap.parse_args()

    log_environment()
    mp4, name = resolve_input(args.input)
    out_dir = Path(".asr_test") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[pi_test] input={mp4.name} name={name}")

    t_all = time.time()

    # 1. audio extract (cached)
    mp3 = out_dir / f"{name}.mp3"
    with ResourceMonitor("convert"):
        a_utils.convert_audio_to_mp3(mp4, mp3)

    # 2. transcribe (cached JSONL)
    transcript = Path("transcripts") / f"{name}.jsonl"
    with ResourceMonitor("transcribe"):
        asr_transcribe.transcribe(mp3, transcript, args.model)

    # 3. detect boundaries (self-hosted hybrid: heuristic + local LLM fallback)
    with ResourceMonitor("detect"):
        result = asr_local.detect(transcript)
    if result is None:
        raise SystemExit("no sermon detected in transcript")

    start = result.start
    end = result.end + END_PADDING_S
    logger.info(f"[pi_test] {name}: {_hms(start)} -> {_hms(end)} "
                f"method={result.method} start_conf={result.start_conf} end_conf={result.end_conf}")

    # 4. trim video
    out_mp4 = out_dir / f"{name}_sermon.mp4"
    with ResourceMonitor("trim"):
        v_utils.trim_video(mp4, out_mp4, int(start), int(end))

    summary = {
        "name": name,
        "source_mp4": mp4.name,
        "start_s": int(start), "end_s": int(end),
        "start_hms": _hms(start), "end_hms": _hms(end),
        "kept_s": int(end) - int(start),
        "method": result.method,
        "start_conf": result.start_conf, "end_conf": result.end_conf,
        "end_padding_s": END_PADDING_S,
        "output": str(out_mp4), "output_exists": out_mp4.exists(),
        "total_seconds": round(time.time() - t_all, 1),
    }
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8")
    (out_dir / "boundary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  {name}: sermon {summary['start_hms']} -> {summary['end_hms']} "
          f"(kept {summary['kept_s'] // 60} min, method={result.method})")
    print(f"  output: {out_mp4}  {'OK' if out_mp4.exists() else 'MISSING'}")
    print(f"  total time: {summary['total_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

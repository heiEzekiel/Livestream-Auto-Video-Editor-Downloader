"""
Manual test harness for the trimming changes (self-enrolled diarization +
percentile-hysteresis trim) against the real services in ``downloaded/``.

For each target date it:
  1. locates the source ``.mp4`` and ``.mp3`` in ``downloaded/<date>/``,
  2. produces per-second similarity scores -- by running the *new* diarization
     (self-enrolled Preacher) or, with ``--reuse``, by loading an existing
     ``similarity_<date>.txt`` (fast: exercises only the trim + ffmpeg path),
  3. computes the sermon span with ``AudioUtils.get_trim_range``,
  4. trims the ``.mp4`` to that span,
  5. writes everything under ``.test/<date>/`` and prints a summary.

Usage (from the repo root, using the project venv):
    python run_trim_test.py                     # every date folder, full diarization
    python run_trim_test.py 2025-12-07          # one date, full diarization
    python run_trim_test.py 2025-12-07 --reuse  # one date, reuse existing scores (fast)

This script is intentionally standalone and is NOT part of the cron pipeline.
"""
import argparse
import sys
import time
from pathlib import Path

from audio_utils import AudioUtils
from video_utils import VideoUtils
from utils import CommonUtils
from configs import Configs
from logger import setup_logger
from resource_monitor import ResourceMonitor, log_environment

cfg = Configs(Path("./config.yaml"))
logger = setup_logger(Path("logs/trim_test.log"), cfg.log_level)

a_utils = AudioUtils()
v_utils = VideoUtils()

DOWNLOADED = Path("./downloaded")
TEST_ROOT = Path("./.test")


def _hms(seconds: float) -> str:
    """Format a second count as H:MM:SS for readable summaries."""
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def find_source_mp4(folder: Path) -> Path | None:
    """Pick the source video in a date folder (largest .mp4 that isn't a trim)."""
    candidates = [
        p for p in folder.glob("*.mp4")
        if "sermon" not in p.name.lower() and "trimmed" not in p.name.lower()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def find_source_mp3(folder: Path, date: str) -> Path | None:
    """Prefer ``<date>.mp3``; otherwise any .mp3 in the folder."""
    preferred = folder / f"{date}.mp3"
    if preferred.exists():
        return preferred
    others = list(folder.glob("*.mp3"))
    return others[0] if others else None


def build_similarity(date: str, folder: Path, out_dir: Path, reuse: bool) -> list[str] | None:
    """
    Return the per-second similarity lines for a date, writing a copy into the
    output dir. Either reuse an existing file or run the new diarization.
    """
    out_sim = out_dir / f"similarity_{date}.txt"

    if reuse:
        # Prefer an already-regenerated copy under .test/, else the source folder.
        candidates = [out_dir / f"similarity_{date}.txt", folder / f"similarity_{date}.txt"]
        existing = next((p for p in candidates if p.exists()), None)
        if existing is None:
            logger.error(f"[{date}] --reuse set but no similarity_{date}.txt found; skipping.")
            return None
        lines = CommonUtils.load_text_file(existing)
        out_sim.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"[{date}] Reused {len(lines)} similarity lines from {existing}.")
        return lines

    mp3 = find_source_mp3(folder, date)
    if mp3 is None:
        # No extracted audio yet -- do what the real pipeline does: mp4 -> mp3.
        mp4 = find_source_mp4(folder)
        if mp4 is None:
            logger.error(f"[{date}] No .mp3 and no source .mp4 in {folder}; cannot diarize.")
            return None
        mp3 = folder / f"{date}.mp3"
        logger.info(f"[{date}] No .mp3 found; extracting audio {mp4.name} -> {mp3.name} ...")
        a_utils.convert_audio_to_mp3(mp4, mp3)

    logger.info(f"[{date}] Running self-enrolled diarization on {mp3.name} ...")
    t0 = time.time()
    speaker_segments = a_utils.speaker_segments_generator(mp3)
    with ResourceMonitor(f"diarization:{date}"):
        per_second = a_utils.run_diarization(mp3, speaker_segments)
    CommonUtils.save_text_file(out_sim, per_second)
    logger.info(f"[{date}] Diarization finished in {_hms(time.time() - t0)} "
                f"({len(per_second)} seconds scored).")
    return CommonUtils.load_text_file(out_sim)


def process_date(date: str, reuse: bool) -> dict | None:
    folder = DOWNLOADED / date
    if not folder.is_dir():
        logger.error(f"[{date}] Folder {folder} does not exist; skipping.")
        return None

    mp4 = find_source_mp4(folder)
    if mp4 is None:
        logger.error(f"[{date}] No source .mp4 in {folder}; skipping.")
        return None

    out_dir = TEST_ROOT / date
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = build_similarity(date, folder, out_dir, reuse)
    if not lines:
        return None

    start, end = a_utils.get_trim_range(lines)
    vid_len = len(lines) - 1

    out_mp4 = out_dir / f"{date}_sermon.mp4"
    logger.info(f"[{date}] Trimming {start}s -> {end}s into {out_mp4.name} ...")
    v_utils.trim_video(mp4, out_mp4, start, end)

    summary = {
        "date": date,
        "source_mp4": mp4.name,
        "video_len_s": vid_len,
        "start_s": start,
        "end_s": end,
        "kept_s": max(0, end - start),
        "output": str(out_mp4),
        "output_exists": out_mp4.exists(),
        "output_size_mb": round(out_mp4.stat().st_size / 1024 / 1024, 1) if out_mp4.exists() else 0,
    }

    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8"
    )
    return summary


def discover_dates() -> list[str]:
    """All ``downloaded/<date>/`` folders that contain a source .mp4."""
    return sorted(
        p.name for p in DOWNLOADED.iterdir()
        if p.is_dir() and find_source_mp4(p) is not None
    )


def main():
    parser = argparse.ArgumentParser(description="Test the video trimming changes.")
    parser.add_argument("dates", nargs="*", help="Date folders to process (default: all).")
    parser.add_argument("--reuse", action="store_true",
                        help="Reuse existing similarity_<date>.txt instead of re-running diarization.")
    args = parser.parse_args()

    dates = args.dates or discover_dates()
    if not dates:
        logger.error("No date folders with a source .mp4 found under downloaded/.")
        sys.exit(1)

    logger.info(f"Testing dates: {', '.join(dates)} (reuse={args.reuse})")
    log_environment()

    results = []
    for date in dates:
        try:
            result = process_date(date, args.reuse)
            if result:
                results.append(result)
        except Exception:
            logger.exception(f"[{date}] Failed to process.")

    print("\n" + "=" * 72)
    print(f"{'DATE':<12}{'VIDEO':>10}{'START':>10}{'END':>10}{'KEPT':>10}  OUTPUT")
    print("-" * 72)
    for r in results:
        print(f"{r['date']:<12}{_hms(r['video_len_s']):>10}{_hms(r['start_s']):>10}"
              f"{_hms(r['end_s']):>10}{_hms(r['kept_s']):>10}  "
              f"{'OK' if r['output_exists'] else 'MISSING'} "
              f"({r['output_size_mb']} MB)")
    print("=" * 72)
    print(f"Outputs written under {TEST_ROOT.resolve()}")


if __name__ == "__main__":
    main()
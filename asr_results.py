"""
Materialize ASR-prototype results into a dedicated, sortable folder.

Outputs live under ``.asr_test/<date>/`` (parallel to the similarity detector's
``.test/<date>/`` so the two approaches can be compared side by side):

    .asr_test/<date>/boundary.json     # machine-readable detected span + metadata
    .asr_test/<date>/summary.txt       # human-readable summary
    .asr_test/<date>/<date>_sermon.mp4  # the source video trimmed to the ASR span

``write_result()`` is the entry point — the boundary source (asr_detect's Claude
API call, or the in-environment validation workflow) is decoupled from how the
result is recorded and trimmed.

CLI:
    python asr_results.py 2026-01-04 3584 8922 [--no-trim] [--note "..."]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from logger import setup_logger
from configs import Configs
from video_utils import VideoUtils
from run_trim_test import find_source_mp4

cfg = Configs(Path("./config.yaml"))
logger = setup_logger(Path("logs/asr.log"), cfg.log_level)
v_utils = VideoUtils()

ASR_TEST = Path("./.asr_test")
DOWNLOADED = Path("./downloaded")


def _hms(x):
    x = int(x)
    return f"{x // 3600}:{(x % 3600) // 60:02d}:{x % 60:02d}"


END_PADDING_S = 5  # seconds added past the detected end so the final words aren't clipped


def write_result(date: str, start: int, end: int, meta: dict | None = None,
                 trim: bool = True, end_padding: int = END_PADDING_S) -> dict:
    """Record an ASR-detected span for ``date`` and (optionally) trim the video.

    ``end`` is the detected end of the preacher-led segment; ``end_padding``
    seconds are added to it before trimming so the last words aren't clipped.
    """
    meta = meta or {}
    out_dir = ASR_TEST / date
    out_dir.mkdir(parents=True, exist_ok=True)

    folder = DOWNLOADED / date
    mp4 = find_source_mp4(folder) if folder.is_dir() else None

    detected_end = int(end)
    final_end = detected_end + int(end_padding)

    record = {
        "date": date,
        "detector": "asr",
        "start_s": int(start),
        "detected_end_s": detected_end,
        "end_padding_s": int(end_padding),
        "end_s": final_end,
        "start_hms": _hms(start),
        "end_hms": _hms(final_end),
        "kept_s": max(0, final_end - int(start)),
        "source_mp4": mp4.name if mp4 else None,
        **meta,
    }

    out_mp4 = out_dir / f"{date}_sermon.mp4"
    if trim and mp4 is not None:
        logger.info(f"[asr:{date}] trimming {start}s -> {final_end}s "
                    f"(detected {detected_end}s + {end_padding}s pad) into {out_mp4.name}")
        v_utils.trim_video(mp4, out_mp4, int(start), final_end)
        record["output"] = str(out_mp4)
        record["output_exists"] = out_mp4.exists()
    else:
        record["output"] = None
        record["output_exists"] = False

    (out_dir / "boundary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in record.items()) + "\n", encoding="utf-8"
    )
    return record


def main():
    ap = argparse.ArgumentParser(description="Write an ASR result into .asr_test/<date>/.")
    ap.add_argument("date")
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    meta = {"note": args.note} if args.note else {}
    rec = write_result(args.date, args.start, args.end, meta=meta, trim=not args.no_trim)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()

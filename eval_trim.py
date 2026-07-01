"""
Evaluation harness for the trim-detection logic.

Scores ``AudioUtils.get_trim_range`` against hand-verified ground truth so that
parameter changes (in ``config.yaml`` under ``trim_logic``) can be judged on
real data instead of by eye. It reads the saved per-second similarity files
(``.test/<date>/similarity_<date>.txt``, falling back to ``downloaded/...``) so
it runs in a second -- no re-diarization needed.

Fill in TRUTH with the actual sermon start/end (seconds into the original
video). Leave a value as ``None`` if unknown; the row is shown but excluded
from the error totals.

Usage:
    python eval_trim.py
"""
from pathlib import Path

from audio_utils import AudioUtils
from utils import CommonUtils

a_utils = AudioUtils()

# date -> {"start": sec|None, "end": sec|None}  (absolute seconds in the source video)
# All values are user-validated against the full sermon recordings.
TRUTH = {
    "2025-12-07": {"start": 3201, "end": 6723},
    "2025-12-14": {"start": 3014, "end": None},
    "2025-12-28": {"start": 3618, "end": None},
    "2026-01-04": {"start": 3584, "end": None},   # start 59:44, end ok
    "2026-01-11": {"start": 4079, "end": None},   # start 67:59 (+2:54 into trim), end ok
    "2026-02-08": {"start": 3346, "end": 8877},   # 55:46 - 2:27:57
    "2026-02-15": {"start": 3445, "end": 7379},   # 57:25 - 2:02:59
    "2026-02-22": {"start": 3682, "end": 6750},   # start ok, end 1:52:30 (51:08 into trim)
    "2026-03-01": {"start": 3197, "end": None},   # correct
    "2026-03-08": {"start": 3859, "end": None},
    "2026-03-22": {"start": 3354, "end": None},   # correct
    "2026-04-05": {"start": 6569, "end": None},   # acceptable as-is (do not pull earlier)
    "2026-04-12": {"start": 3024, "end": None},   # start 50:24, end ok
    "2026-04-19": {"start": 3686, "end": None},   # start 61:26 (+2:27 into trim), end ok
    "2026-04-26": {"start": 3128, "end": None},   # start 52:08, end ok
    "2026-05-03": {"start": 3632, "end": None},   # correct
    "2026-05-10": {"start": 3508, "end": None},
    "2026-05-24": {"start": 3384, "end": None},
    "2026-05-31": {"start": 3537, "end": None},
}


def _similarity_file(date: str) -> Path | None:
    for p in (Path(f".test/{date}/similarity_{date}.txt"),
              Path(f"downloaded/{date}/similarity_{date}.txt")):
        if p.exists():
            return p
    return None


def _hms(x):
    if x is None:
        return "   ?   "
    x = int(x)
    return f"{x // 3600}:{(x % 3600) // 60:02d}:{x % 60:02d}"


def main():
    print(f"{'DATE':<12}{'PRED start':>11}{'TRUTH':>10}{'d_start':>8}   "
          f"{'PRED end':>11}{'TRUTH':>10}{'d_end':>8}")
    print("-" * 74)

    start_errs, end_errs = [], []
    for date, truth in TRUTH.items():
        f = _similarity_file(date)
        if f is None:
            print(f"{date:<12}  (no similarity file found)")
            continue

        lines = CommonUtils.load_text_file(f)
        start, end = a_utils.get_trim_range(lines)

        ds = "" if truth["start"] is None else f"{start - truth['start']:+d}s"
        de = "" if truth["end"] is None else f"{end - truth['end']:+d}s"
        if truth["start"] is not None:
            start_errs.append(abs(start - truth["start"]))
        if truth["end"] is not None:
            end_errs.append(abs(end - truth["end"]))

        print(f"{date:<12}{_hms(start):>11}{_hms(truth['start']):>10}{ds:>8}   "
              f"{_hms(end):>11}{_hms(truth['end']):>10}{de:>8}")

    print("-" * 74)
    if start_errs:
        print(f"start error: n={len(start_errs)}  mean={sum(start_errs)/len(start_errs):.0f}s  "
              f"max={max(start_errs)}s")
    if end_errs:
        print(f"end   error: n={len(end_errs)}  mean={sum(end_errs)/len(end_errs):.0f}s  "
              f"max={max(end_errs)}s")


if __name__ == "__main__":
    main()
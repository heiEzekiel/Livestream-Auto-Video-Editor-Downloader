"""
Evaluate the ASR sermon-boundary detector against the validated ground truth.

Runs asr_detect.detect() on every date that has both a transcript
(transcripts/<date>.jsonl) and a TRUTH entry in eval_trim, then reports the
start/end error vs the user-validated timestamps. This is how we judge whether
the content-based detector beats the acoustic similarity detector on the
semantic cases — on the same benchmark, with the same numbers.

    python asr_eval.py                 # all dates with a transcript + truth
    python asr_eval.py 2026-01-04 ...  # specific dates
    python asr_eval.py --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval_trim import TRUTH
import asr_detect

TRANSCRIPTS = Path("./transcripts")


def _hms(x):
    if x is None:
        return "   ?   "
    x = int(x)
    return f"{x // 3600}:{(x % 3600) // 60:02d}:{x % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description="Score ASR detector vs ground truth.")
    ap.add_argument("dates", nargs="*")
    ap.add_argument("--model", default=asr_detect.DEFAULT_MODEL)
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    dates = args.dates or sorted(
        p.stem for p in TRANSCRIPTS.glob("*.jsonl") if p.stem in TRUTH
    )
    if not dates:
        print("No transcripts with ground truth found. Run asr_transcribe.py first.")
        return

    print(f"model={args.model}\n")
    print(f"{'DATE':<12}{'PRED start':>11}{'TRUTH':>10}{'d_start':>9}   "
          f"{'PRED end':>11}{'TRUTH':>10}{'d_end':>9}  ovr")
    print("-" * 80)

    start_errs, end_errs = [], []
    for date in dates:
        path = TRANSCRIPTS / f"{date}.jsonl"
        if not path.exists():
            print(f"{date:<12}  (no transcript)")
            continue
        truth = TRUTH.get(date, {})
        try:
            r = asr_detect.detect(path, client=client, model=args.model)
        except Exception as e:
            print(f"{date:<12}  ERROR: {e}")
            continue

        ts, te = truth.get("start"), truth.get("end")
        ds = "" if ts is None else f"{r.start - ts:+d}s"
        de = "" if te is None else f"{r.end - te:+d}s"
        if ts is not None:
            start_errs.append(abs(r.start - ts))
        if te is not None:
            end_errs.append(abs(r.end - te))

        print(f"{date:<12}{_hms(r.start):>11}{_hms(ts):>10}{ds:>9}   "
              f"{_hms(r.end):>11}{_hms(te):>10}{de:>9}  {'Y' if r.agreed else 'N'}")

    print("-" * 80)
    if start_errs:
        print(f"start error: n={len(start_errs)}  mean={sum(start_errs)/len(start_errs):.0f}s  "
              f"max={max(start_errs)}s  within_30s={sum(1 for e in start_errs if e <= 30)}/{len(start_errs)}")
    if end_errs:
        print(f"end   error: n={len(end_errs)}  mean={sum(end_errs)/len(end_errs):.0f}s  "
              f"max={max(end_errs)}s  within_30s={sum(1 for e in end_errs if e <= 30)}/{len(end_errs)}")


if __name__ == "__main__":
    main()

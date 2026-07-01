"""
Local (self-hosted) LLM boundary refinement — the second half of the hybrid
detector (asr_local.py). Used ONLY for low-confidence boundaries (no spoken
handoff, multi-sermon services), so it runs rarely. Runs a small quantized
instruct model on-device via llama-cpp-python (CPU) — no cloud API.

Only the window around the uncertain boundary (±``WINDOW_S`` seconds, centered on
the heuristic candidate) is down-sampled to ~one line per ``BIN_S`` seconds and
sent — the confident boundary stays the heuristic anchor and is never sent — so
the prompt stays small and fast on a Raspberry Pi. (The rare both-uncertain
service falls back to the full transcript.) The model returns the sermon
start/end in seconds; the caller substitutes only the uncertain one.

Model: a GGUF placed at ``models/`` (default Qwen2.5-3B-Instruct Q4_K_M).
Override with the ASR_LLM_MODEL env var.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

BIN_S = 20
# Only the UNCERTAIN boundary is sent to the model, as a window CENTERED on the
# heuristic candidate with this much margin each side. A generous margin keeps the
# true boundary inside the window even when the heuristic candidate is minutes off
# (e.g. the no-handoff start, anchored at the region's left edge). Env-overridable.
WINDOW_S = int(os.environ.get("ASR_REFINE_WINDOW_S", "420"))  # ±7 min
MODEL_PATH = Path(os.environ.get(
    "ASR_LLM_MODEL", "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"))

_SYSTEM = (
    "You locate the main sermon in a church-service transcript. The service runs: "
    "pre-service, worship songs, welcome, announcements, sometimes a video/special "
    "item, then the PREACHER'S SEGMENT (the message), then closing (altar call / "
    "prayer / announcements / dismissal).\n"
    "START = where the service is handed to the preacher (host welcoming the preacher, "
    "or the preacher taking the stage). If a worship song flows directly into the "
    "preacher's opening with no spoken handoff, start at that final lead-in song. "
    "Include preacher-led scripture recitation/ministry; do NOT wait for expository "
    "teaching. NOT the earlier announcements/worship.\n"
    "END = after the preacher-led closing prayer/altar call, before post-service "
    "announcements / a final worship-team song / dismissal.\n"
    "Use the [N] second markers. Reply with ONLY JSON: "
    '{"start_seconds": <int>, "end_seconds": <int>}.'
)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"local LLM model not found: {MODEL_PATH}")
        from llama_cpp import Llama
        # n_ctx: the KV cache is allocated for the full n_ctx once, regardless of
        #   prompt length. Windowed refines use a few k tokens; the only large
        #   prompt is the rare both-uncertain full-transcript fallback (~14-18k),
        #   so 20480 keeps RAM/alloc down while never truncating that fallback.
        # n_threads: default to all cores (the measured ~75C is below the Pi 5's
        #   ~80-85C throttle, so 4 threads is not throttling). ASR_LLM_THREADS lets
        #   the operator A/B 3-vs-4 for a cooler-vs-faster trade on the real unit.
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=int(os.environ.get("ASR_LLM_NCTX", "20480")),
            n_threads=int(os.environ.get("ASR_LLM_THREADS", str(os.cpu_count() or 4))),
            verbose=False,
        )
    return _llm


def window_segs(segs: list[dict], center_s: int, window_s: int = WINDOW_S) -> list[dict]:
    """Segments whose start is within ``±window_s`` of ``center_s``.

    The margin is deliberately generous so the true boundary stays inside the
    window even when the heuristic candidate (``center_s``) is off by minutes.
    """
    lo, hi = center_s - window_s, center_s + window_s
    return [s for s in segs if lo <= int(s["start"]) <= hi]


def downsample(segs: list[dict], bin_s: int = BIN_S, max_lines: int = 900) -> str:
    """Merge segments into ~bin_s-second lines: '[start_sec] text text ...'."""
    bins: dict[int, list[str]] = {}
    for s in segs:
        b = (int(s["start"]) // bin_s) * bin_s
        txt = (s.get("text") or "").strip()
        if txt:
            bins.setdefault(b, []).append(txt)
    lines = [f"[{b}] {' '.join(v)}" for b, v in sorted(bins.items())]
    if len(lines) > max_lines:  # keep it within the small-model context
        step = len(lines) / max_lines
        lines = [lines[int(i * step)] for i in range(max_lines)]
    return "\n".join(lines)


def _parse(text: str) -> tuple[int, int] | None:
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return int(d["start_seconds"]), int(d["end_seconds"])
    except (ValueError, KeyError, TypeError):
        return None


def refine_boundaries(segs: list[dict], heuristic) -> tuple[int, int]:
    """
    Return (start_s, end_s) for the sermon. The high-confidence heuristic
    boundary is passed as an anchor and kept; the model resolves the rest.
    Falls back to the heuristic values if the model output can't be parsed.
    """
    llm = _get_llm()
    anchors = []
    if heuristic.start_conf >= 0.7:
        anchors.append(f"The start is known to be ~{heuristic.start_s} seconds.")
    if heuristic.end_conf >= 0.7:
        anchors.append(f"The end is known to be ~{heuristic.end_s} seconds.")
    anchor_txt = (" ".join(anchors) + "\n") if anchors else ""

    # Send the model only what it needs: a window around the UNCERTAIN boundary
    # (the confident one stays the heuristic anchor and is never sent). This is
    # the big Pi win — prefill is linear in prompt length and dominates the CPU
    # cost; a ±WINDOW_S window is a small fraction of a ~2.5 h transcript. Only
    # the rare both-uncertain service falls back to the full transcript.
    if heuristic.start_conf < 0.7 and heuristic.end_conf < 0.7:
        win = segs                                          # both uncertain
    elif heuristic.start_conf < 0.7:
        win = window_segs(segs, heuristic.start_s)          # uncertain START
    else:
        win = window_segs(segs, heuristic.end_s)            # uncertain END
    user = (anchor_txt + "Transcript (second | text):\n" + downsample(win))
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=120,
    )
    parsed = _parse(out["choices"][0]["message"]["content"])
    if parsed is None:
        return heuristic.start_s, heuristic.end_s
    start, end = parsed
    # keep the boundary the heuristic was already confident about
    if heuristic.start_conf >= 0.7:
        start = heuristic.start_s
    if heuristic.end_conf >= 0.7:
        end = heuristic.end_s
    if not (0 <= start < end):              # reject a nonsensical model span
        return heuristic.start_s, heuristic.end_s
    return start, end

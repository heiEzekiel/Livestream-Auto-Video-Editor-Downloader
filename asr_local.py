"""
Self-hosted sermon-boundary detection from a transcript (DESIGN.md §8a) — no
cloud API. Hybrid design:

  1. heuristic_detect()  — fast, deterministic, in-app. Handoff-phrase detection
     ("welcome Pastor X", "preaching of the word") + lexical scoring (scripture
     refs / sermon vocabulary vs announcement / worship words) to find the
     preacher's segment. Returns boundaries WITH confidence + evidence.
  2. detect()            — orchestrates: trust the heuristic when confident;
     fall back to a local LLM (asr_llm.refine_boundaries) for any low-confidence
     boundary. The LLM is sent only a ±WINDOW_S window centered on the uncertain
     boundary (the confident side stays the heuristic anchor and is never sent),
     so the prompt — and CPU-bound prefill — stays small on a Pi.

This module is pure-Python and unit-testable; the optional local-LLM step is
imported lazily so the heuristic path has zero extra dependencies.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from asr_detect import load_transcript  # reuse JSONL loader

logger = logging.getLogger("sermon_pipeline")

# ----------------------------------------------------------------------------
# Lexicons
# ----------------------------------------------------------------------------
# Strong "service handed to the preacher" cues — the most reliable START marker.
HANDOFF_PATTERNS = [
    r"preaching of the word",
    r"\b(ready|are you ready)\b.{0,20}\b(preaching|word|message|sermon)\b",
    r"\bhelp me\b.{0,20}\b(welcome|invite)\b",
    r"\b(let'?s|please)\b.{0,30}\b(welcome|invite|receive)\b.{0,30}\b(pastor|ps\.?|pr\.?|preacher)\b",
    r"\b(welcome|invite|receive)\b.{0,30}\bfor the (preaching|message|word|sermon)\b",
    r"\bwelcome\b.{0,25}\b(pastor|ps\.?|pr\.?)\b",
    r"\bwelcome (our|back our|our beloved)\b.{0,25}\b(pastor|preacher)\b",
    r"\bhand (it )?over\b.{0,25}\b(pastor|preacher)\b",
    r"\binvite\b.{0,20}\b(pastor|ps\.?)\b",
]

# 66 books (lowercase, ordinal prefixes handled separately as "corinthians" etc.)
SCRIPTURE_BOOKS = {
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy", "joshua", "judges",
    "ruth", "samuel", "kings", "chronicles", "ezra", "nehemiah", "esther", "job",
    "psalm", "psalms", "proverbs", "ecclesiastes", "isaiah", "jeremiah",
    "lamentations", "ezekiel", "daniel", "hosea", "joel", "amos", "obadiah",
    "jonah", "micah", "nahum", "habakkuk", "zephaniah", "haggai", "zechariah",
    "malachi", "matthew", "mark", "luke", "john", "acts", "romans", "corinthians",
    "galatians", "ephesians", "philippians", "colossians", "thessalonians",
    "timothy", "titus", "philemon", "hebrews", "james", "peter", "jude",
    "revelation",
}
SERMON_WORDS = {
    "grace", "righteous", "righteousness", "gospel", "jesus", "christ", "lord",
    "faith", "scripture", "bible", "verse", "chapter", "heaven", "sin", "sins",
    "salvation", "blessing", "anointing", "covenant", "cross", "father", "preach",
    "redemption", "mercy", "the word", "god's", "holy spirit", "believe",
}
ANNOUNCE_PHRASES = {
    "sign up", "signup", "register", "registration", "this week", "next week",
    "this coming", "e-card", "information counter", "cell group", "care group",
    "rsvp", "ticket", "website", "announcement", "announcements", "upcoming",
    "save the date", "mark your calendar", "volunteer", "do join us", "do sign up",
    "happening on", "next sunday", "orientation",
}
# Preacher-led closing cues (altar call / salvation prayer / benediction).
CLOSING_PHRASES = {
    "every head bowed", "every eye closed", "say this prayer", "pray this prayer",
    "receive jesus", "receive him", "receive you as my", "altar", "lift up your hands",
    "the lord bless you", "bless you and keep you", "benediction", "in jesus name",
    "in jesus' name", "welcome to the family", "salvation prayer", "god bless you",
}
# Post-service / dismissal cues — the END must stop before these.
DISMISSAL_PHRASES = {
    "see you next", "see you next sunday", "thank you for joining", "next sunday",
    "have a blessed week", "thanks for joining", "join us next",
}

_WORD = re.compile(r"[a-z']+")
_SCRIPTURE_REF = re.compile(r"\b([1-3]\s+)?[a-z]+\s+(chapter\s+)?\d{1,3}(:\d{1,3})?\b")


def _count_phrases(text: str, phrases) -> int:
    return sum(1 for p in phrases if p in text)


def is_handoff(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in HANDOFF_PATTERNS)


def sermon_score(text: str) -> float:
    """Per-segment score: positive = sermon-like, negative = announcement-like."""
    t = text.lower()
    words = _WORD.findall(t)
    wset = set(words)
    score = 0.0
    score += 2.0 * len(SCRIPTURE_BOOKS & wset)          # scripture book mentions
    if _SCRIPTURE_REF.search(t):
        score += 1.5                                     # "John 3:16" / "Psalm 23"
    score += sum(0.5 for w in SERMON_WORDS if w in t)
    score -= 1.5 * _count_phrases(t, ANNOUNCE_PHRASES)   # announcements push negative
    return score


# ----------------------------------------------------------------------------
# Region detection
# ----------------------------------------------------------------------------
def _smooth(values, window: int):
    if window <= 1 or not values:
        return list(values)
    n = len(values)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def longest_sermon_region(scores, thr: float, gap: int, min_len: int):
    """Longest run of indices with smoothed score >= thr, bridging gaps."""
    above = [s >= thr for s in scores]
    idx = [i for i, a in enumerate(above) if a]
    if not idx:
        return None
    best = None
    start = prev = idx[0]
    for x in idx[1:]:
        if x - prev <= gap + 1:
            prev = x
        else:
            if prev - start + 1 >= min_len and (best is None or prev - start > best[1] - best[0]):
                best = (start, prev)
            start = prev = x
    if prev - start + 1 >= min_len and (best is None or prev - start > best[1] - best[0]):
        best = (start, prev)
    return best


@dataclass
class HeuristicResult:
    start_idx: int
    end_idx: int
    start_s: int
    end_s: int
    start_conf: float          # 0..1
    end_conf: float
    evidence: dict = field(default_factory=dict)


def heuristic_detect(segs: list[dict]) -> HeuristicResult | None:
    """
    Deterministic boundary detection.

    START: the *last* genuine preacher-handoff phrase in the plausible window
    (first 2/3 of the service, after ~20 min), excluding handoffs that introduce
    the *announcer* ("welcome ... for the announcements"). High confidence when a
    handoff is found; low (-> defer to the local LLM) when none is — exactly the
    no-spoken-handoff services (worship segues straight into the message, or a
    multi-sermon service).

    END: the last preacher-led closing cue (altar call / salvation prayer /
    benediction) before a post-service dismissal cue.
    """
    if not segs:
        return None
    secs = [int(s["start"]) for s in segs]
    duration = secs[-1] if secs else 0

    # rough sermon body, for END localization + confidence
    sm = _smooth([sermon_score(s.get("text", "")) for s in segs], 9)
    pos = [s for s in sm if s > 0]
    thr = (sum(pos) / len(pos) * 0.5) if pos else 0.5
    region = longest_sermon_region(sm, thr, gap=25, min_len=20)
    r0, r1 = region if region else (0, len(segs) - 1)

    # ---- START: last real preacher handoff in [20min, 2/3 of service] ----
    lo, hi = 20 * 60, max(20 * 60, int(duration * 0.66))
    handoff_idxs = [
        i for i, s in enumerate(segs)
        if lo <= secs[i] <= hi
        and is_handoff(s.get("text", ""))
        and "announcement" not in s.get("text", "").lower()
    ]
    if handoff_idxs:
        start_idx = handoff_idxs[-1]                 # last genuine handoff
        start_conf = 0.85
    else:
        start_idx = r0                               # no handoff -> defer to LLM
        start_conf = 0.4

    # ---- END: last closing cue before dismissal, near/after the region ----
    end_idx = r1
    end_conf = 0.5
    closing_hits = []
    scan_from = min(r1, len(segs) - 1)
    for i in range(scan_from, len(segs)):
        t = segs[i].get("text", "").lower()
        if _count_phrases(t, CLOSING_PHRASES):
            closing_hits.append(i)
        elif closing_hits and _count_phrases(t, DISMISSAL_PHRASES):
            break
    if closing_hits:
        end_idx = closing_hits[-1]
        end_conf = 0.8

    return HeuristicResult(
        start_idx=start_idx, end_idx=end_idx,
        start_s=secs[start_idx], end_s=secs[end_idx],
        start_conf=start_conf, end_conf=end_conf,
        evidence={
            "region": (secs[r0], secs[r1]),
            "handoff_text": segs[start_idx].get("text", "")[:90] if handoff_idxs else None,
            "closing_text": segs[end_idx].get("text", "")[:90] if closing_hits else None,
            "n_handoffs": len(handoff_idxs),
        },
    )


@dataclass
class LocalResult:
    start: int
    end: int
    start_conf: float
    end_conf: float
    method: str            # "heuristic" or "heuristic+llm"
    evidence: dict = field(default_factory=dict)


def detect(path, conf_threshold: float = 0.7, use_llm: bool | None = None) -> LocalResult | None:
    """
    Self-hosted detection: heuristic first; for a boundary below
    ``conf_threshold`` confidence, refine with the local LLM over a short window
    centered on that boundary (asr_llm.refine_boundaries).

    ``use_llm`` defaults to the SKIP_LLM environment variable (SKIP_LLM=1 ->
    heuristic only). Returns None if no usable span is found (caller falls back).
    """
    if use_llm is None:
        use_llm = os.environ.get("SKIP_LLM", "0") != "1"

    segs = load_transcript(path)
    h = heuristic_detect(segs)
    if h is None:
        return None

    start, end = h.start_s, h.end_s
    method = "heuristic"
    if use_llm and (h.start_conf < conf_threshold or h.end_conf < conf_threshold):
        try:
            from asr_llm import refine_boundaries
            ls, le = refine_boundaries(segs, h)
            if 0 <= ls < le:                 # only trust a sane LLM span
                start, end, method = ls, le, "heuristic+llm"
            else:
                logger.warning("local LLM returned an invalid span (%s-%s); keeping heuristic", ls, le)
        except Exception:  # local LLM unavailable -> keep heuristic
            pass

    if not (0 <= start < end):               # heuristic span itself is unusable
        return None

    return LocalResult(
        start=int(start), end=int(end),
        start_conf=h.start_conf, end_conf=h.end_conf,
        method=method, evidence=h.evidence,
    )

"""
ASR-based sermon boundary detector (DESIGN.md §5).

The acoustic similarity detector hit a ceiling on *semantic* cases: pre-sermon
worship/testimony scores as high against the preacher reference as the sermon
itself, and a service can contain preacher-led segments (communion, Q&A) that
aren't the message. Those distinctions live in the *content*, not the audio.

This module reads a timestamped transcript (from asr_transcribe.py) and asks
Claude to identify where the main sermon/message starts and ends, then runs an
independent adversarial verification pass that tries to move each boundary. It
returns a structured result with the timestamps, the quotes they correspond to,
and the model's reasoning.

Production note: this is a single Claude API call per service (plus one
verification call) on a weekly job — cheap and well within a Pi's reach (the
heavy ASR step is local). Model defaults to Opus 4.8; pass --model claude-sonnet-4-6
for a cheaper run.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from logger import setup_logger
from configs import Configs

cfg = Configs(Path("./config.yaml"))
logger = setup_logger(Path("logs/asr.log"), cfg.log_level)

TRANSCRIPTS = Path("./transcripts")
DEFAULT_MODEL = "claude-opus-4-8"

# ----------------------------------------------------------------------------
# Structured-output schema (Claude returns exactly this shape)
# ----------------------------------------------------------------------------
_BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "start_seconds": {
            "type": "integer",
            "description": "Absolute second the service is HANDED OVER TO THE PREACHER for the "
                           "message (a host introducing/welcoming the preacher, or the preacher "
                           "taking the stage), after pre-service, worship songs, welcome, and "
                           "announcements. INCLUDE preacher-led scripture recitation/declaration/"
                           "ministry that leads into the teaching; do NOT wait for the expository "
                           "teaching to begin.",
        },
        "end_seconds": {
            "type": "integer",
            "description": "Absolute second when the preacher-led portion ends. INCLUDE the "
                           "closing prayer / altar call / ministry time the preacher leads "
                           "after the teaching. STOP before post-service announcements, a "
                           "final worship-team song set, or dismissal. Brief worship or prayer "
                           "interludes WITHIN the message are part of the span.",
        },
        "start_quote": {"type": "string", "description": "The transcript line at the start."},
        "end_quote": {"type": "string", "description": "The transcript line at the end."},
        "reasoning": {"type": "string", "description": "Brief justification (2-4 sentences)."},
    },
    "required": ["start_seconds", "end_seconds", "start_quote", "end_quote", "reasoning"],
    "additionalProperties": False,
}

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "agree": {"type": "boolean", "description": "True if the proposed boundaries are correct."},
        "start_seconds": {"type": "integer", "description": "Corrected start (same as proposed if agree)."},
        "end_seconds": {"type": "integer", "description": "Corrected end (same as proposed if agree)."},
        "issue": {"type": "string", "description": "What was wrong, or 'none'."},
    },
    "required": ["agree", "start_seconds", "end_seconds", "issue"],
    "additionalProperties": False,
}

_DETECT_SYSTEM = (
    "You analyze transcripts of church services to locate the main sermon (the "
    "preacher's message/teaching). A service runs roughly: pre-service, worship "
    "songs, welcome, announcements, sometimes a video or special item, then the "
    "SERMON, then closing (altar call/prayer, announcements, dismissal).\n\n"
    "Identify the preacher's segment as ONE contiguous span. The START is the moment "
    "the service is HANDED OVER TO THE PREACHER for the message — a host introducing/"
    "welcoming the preacher ('help me welcome Pastor X', 'are you ready for the Word?') "
    "or the preacher taking the stage. INCLUDE any preacher-led scripture recitation, "
    "declaration, or worship/ministry that is part of his segment leading into the "
    "teaching; do NOT wait for the expository teaching to begin. The start is NOT the "
    "earlier announcements or the congregational worship set before the handoff. Brief "
    "worship/scripture/prayer interludes WITHIN the segment are part of it — do not "
    "split on them. The END includes the closing prayer / altar call / ministry the "
    "preacher leads after the teaching, and stops before post-service announcements, a "
    "final worship-team song set, or dismissal. Use the [NNNN] second markers for timestamps."
)

_VERIFY_SYSTEM = (
    "You are an adversarial reviewer checking proposed sermon start/end timestamps "
    "against a church-service transcript. Try to disprove them. START is when the service "
    "is handed to the PREACHER (host welcoming the preacher / preacher taking the stage), "
    "INCLUDING preacher-led scripture recitation/ministry before the teaching — not where "
    "the expository teaching starts. Is the start too late (it skipped the handoff and a "
    "preacher-led recitation/ministry that should be included)? Is it too early (it lands "
    "in announcements or the pre-handoff worship set)? Does the end "
    "stop too early (cutting off the preacher-led closing prayer/altar call, which "
    "should be kept) or too late (running into post-service announcements / a final "
    "worship set / dismissal)? Brief worship/prayer/scripture WITHIN the message "
    "belongs inside the span. Only set agree=false if you would move a boundary by "
    "more than ~30 seconds; return corrected timestamps."
)


# ----------------------------------------------------------------------------
# Transcript IO (pure, testable)
# ----------------------------------------------------------------------------
def load_transcript(path: Path) -> list[dict]:
    """Load a JSONL transcript into a list of {start, end, text} dicts."""
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segs.append(json.loads(line))
    return segs


def format_transcript(segs: list[dict]) -> str:
    """Render segments as '[NNNN] text' lines (absolute seconds)."""
    return "\n".join(f"[{int(s['start'])}] {s['text']}" for s in segs if s.get("text"))


def _extract_json(message) -> dict:
    """Pull the JSON object out of a structured-output message."""
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        raise ValueError("No text block in response")
    return json.loads(text)


# ----------------------------------------------------------------------------
# LLM calls
# ----------------------------------------------------------------------------
def _call(client, model, system, schema, user_text, max_tokens=16000):
    """One structured, adaptive-thinking, streamed call returning the parsed object."""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user_text}],
    ) as stream:
        message = stream.get_final_message()
    return _extract_json(message), message.usage


def identify_sermon(segs: list[dict], client, model: str = DEFAULT_MODEL) -> tuple[dict, object]:
    body = ("Here is the timestamped transcript of a church service. Identify the "
            "main sermon's start and end.\n\n" + format_transcript(segs))
    return _call(client, model, _DETECT_SYSTEM, _BOUNDARY_SCHEMA, body)


def verify_boundary(segs: list[dict], candidate: dict, client, model: str = DEFAULT_MODEL):
    body = (
        f"Proposed sermon span: start={candidate['start_seconds']}s "
        f"(\"{candidate['start_quote']}\"), end={candidate['end_seconds']}s "
        f"(\"{candidate['end_quote']}\").\n\n"
        "Review against the transcript below and correct if needed.\n\n"
        + format_transcript(segs)
    )
    return _call(client, model, _VERIFY_SYSTEM, _VERDICT_SCHEMA, body)


@dataclass
class AsrResult:
    start: int
    end: int
    proposed_start: int
    proposed_end: int
    agreed: bool
    issue: str
    reasoning: str


def detect(path: Path, client=None, model: str = DEFAULT_MODEL) -> AsrResult:
    """Full pipeline: load → identify → adversarially verify → final boundaries."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    segs = load_transcript(path)
    proposal, u1 = identify_sermon(segs, client, model)
    logger.info(f"[{path.stem}] proposed {proposal['start_seconds']}-{proposal['end_seconds']}s "
                f"| {proposal['reasoning']}")

    verdict, u2 = verify_boundary(segs, proposal, client, model)
    final_start = int(verdict["start_seconds"])
    final_end = int(verdict["end_seconds"])
    logger.info(f"[{path.stem}] verify agree={verdict['agree']} "
                f"final {final_start}-{final_end}s | {verdict['issue']}")

    return AsrResult(
        start=final_start, end=final_end,
        proposed_start=int(proposal["start_seconds"]), proposed_end=int(proposal["end_seconds"]),
        agreed=bool(verdict["agree"]), issue=verdict["issue"], reasoning=proposal["reasoning"],
    )


def _hms(x):
    x = int(x)
    return f"{x // 3600}:{(x % 3600) // 60:02d}:{x % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description="Detect sermon boundaries from a transcript.")
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    print(f"{'DATE':<12}{'START':>10}{'END':>10}  agree  issue")
    print("-" * 70)
    for date in args.dates:
        path = TRANSCRIPTS / f"{date}.jsonl"
        if not path.exists():
            print(f"{date:<12}  (no transcript)")
            continue
        try:
            r = detect(path, client, args.model)
            print(f"{date:<12}{_hms(r.start):>10}{_hms(r.end):>10}  "
                  f"{'Y' if r.agreed else 'N':>5}  {r.issue[:40]}")
        except Exception as e:
            logger.exception(f"[{date}] detect failed")
            print(f"{date:<12}  ERROR: {e}")


if __name__ == "__main__":
    main()
